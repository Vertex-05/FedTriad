# auror_defense.py
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
import warnings
import time
import torch
from copy import deepcopy

class AurorDefense:
    def __init__(self, alpha=0.02, tau=0.5, epochs_to_analyze=10, verbose=True, pca_dim=8):
        """
        alpha: 聚类中心距离阈值
        tau: 恶意用户判断阈值（比例）
        epochs_to_analyze: 时间窗口（目前未使用多轮历史，保留接口）
        verbose: 是否打印调试信息
        pca_dim: 聚类前 PCA 降到的维度（当原始特征维过大时）
        """
        self.alpha = alpha
        self.tau = tau
        self.epochs_to_analyze = epochs_to_analyze
        self.verbose = verbose
        self.pca_dim = pca_dim

    # ================= 特征提取：将模型参数映射到低维稳定特征 =================
    def extract_per_layer_stats(self, model):
        """
        提取每一层的统计量作为特征，避免直接 flatten 全部参数（太高维）。
        返回: 1D np.array, shape = (n_layers * n_stats,)
        统计量: L2 norm, mean, std per layer -> 3 * n_layers
        """
        stats = []
        # 只遍历可学习的参数（按 module 顺序），每个 param 看作一个“层”
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            arr = p.detach().cpu().numpy().ravel()
            if arr.size == 0:
                stats.extend([0.0, 0.0, 0.0])
            else:
                stats.append(float(np.linalg.norm(arr)))   # L2
                stats.append(float(np.mean(arr)))          # mean
                stats.append(float(np.std(arr)))           # std
        return np.array(stats, dtype=np.float32)

    # ================== 主流程 defend ==================
    def defend(self, all_client_models, train_global_model_func=None):
        """
        all_client_models: list of client.model (torch.nn.Module) for participating clients in this round
        train_global_model_func: optional function(trainable_indices) -> new global model,
                                we don't call it inside unless provided and necessary.
        返回: malicious_users(list of indices), indicative_features(list of indices)
        """
        t0 = time.time()
        n_clients = len(all_client_models)
        if self.verbose:
            print(f"[Auror] defend called: n_clients={n_clients}, alpha={self.alpha}, tau={self.tau}")

        # 1) 提取每个客户端的低维特征向量
        X_list = []
        for i, model in enumerate(all_client_models):
            stats = self.extract_per_layer_stats(model)
            X_list.append(stats)
        X = np.stack(X_list, axis=0)  # shape (n_clients, n_features)
        if self.verbose:
            print(f"[Auror] extracted features shape: {X.shape}")

        # 2) 如果特征全常数或维度太小，直接返回“无检测”
        if X.shape[1] == 0 or np.allclose(X, X[0, :]):
            if self.verbose:
                print("[Auror] all features identical across clients -> no indicative features")
            return [], []

        # 3) 可选 PCA 降维（当维度高于 pca_dim）
        X_for_clustering = X
        if X.shape[1] > self.pca_dim:
            try:
                pca = PCA(n_components=min(self.pca_dim, X.shape[1]))
                X_for_clustering = pca.fit_transform(X)
                if self.verbose:
                    print(f"[Auror] applied PCA: new shape {X_for_clustering.shape}")
            except Exception as e:
                if self.verbose:
                    print(f"[Auror] PCA failed: {e}, fallback to raw features")

        # 4) 识别“指示性特征”：这里我们用每个原始特征维度（不是 PCA 后）判断是否存在明显双峰分离。
        indicative_features = []
        n_features = X.shape[1]
        for feature_idx in range(n_features):
            vec = X[:, feature_idx]
            # 跳过常数/方差极小特征
            if np.std(vec) < 1e-8:
                continue
            # 对该维度做 1-d KMeans (2 clusters)，但要保护可能的异常
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
                    labels = kmeans.fit_predict(vec.reshape(-1, 1))
                    centers = kmeans.cluster_centers_.reshape(-1)
                dist = abs(centers[0] - centers[1])
                if dist > self.alpha:
                    indicative_features.append(feature_idx)
            except Exception as e:
                # 若 KMeans 失败（少样本或其他问题），跳过该维度
                if self.verbose:
                    print(f"[Auror] feature {feature_idx} KMeans failed: {e}")
                continue

        if self.verbose:
            print(f"[Auror] indicative_features count: {len(indicative_features)} / {n_features}")

        # 若没有指示性特征 -> 无法检测
        if len(indicative_features) == 0:
            if self.verbose:
                print("[Auror] no indicative features found -> returning no malicious users")
            return [], indicative_features

        # 5) 对每个指示性特征做聚类并统计可疑用户出现次数
        suspicion_counts = np.zeros(n_clients, dtype=int)
        for feature_idx in indicative_features:
            vec = X[:, feature_idx].reshape(-1, 1)
            # 先尝试 KMeans；若失败，退化到 AgglomerativeClustering（较稳健）
            labels = None
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
                    labels = kmeans.fit_predict(vec)
            except Exception:
                # fallback
                try:
                    agg = AgglomerativeClustering(n_clusters=2)
                    labels = agg.fit_predict(vec)
                except Exception as e:
                    if self.verbose:
                        print(f"[Auror] clustering failed on feature {feature_idx}: {e}")
                    continue

            # 哪个簇为“少数”簇（较可疑）
            c0 = int(np.sum(labels == 0))
            c1 = int(np.sum(labels == 1))
            suspicious_cluster = 0 if c0 < c1 else 1
            suspicion_counts += (labels == suspicious_cluster).astype(int)

        # 6) 判定恶意用户：出现在 > tau * n_indicative_feature 次的用户
        threshold = max(1, int(np.ceil(self.tau * len(indicative_features))))
        malicious_users = [i for i, c in enumerate(suspicion_counts) if c >= threshold]

        if self.verbose:
            print(f"[Auror] suspicion_counts: {suspicion_counts.tolist()}")
            print(f"[Auror] threshold: {threshold}, malicious_users detected: {malicious_users}")
            print(f"[Auror] total time: {time.time() - t0:.3f}s")

        # 返回：恶意用户索引 & 指示性特征（便于调试/可视化）
        return malicious_users, indicative_features
