import torch
import numpy as np
from copy import deepcopy

def aggregate_clients_with_krum(
    participating_clients,
    server_model,
    f,
    select_ratio=0.66,     # 默认选中 2/3 客户端
    device='cpu',
    overall_label=None,    # True=良性, False=恶意（来自 run_ours）
    total_clients=None     # 总客户端数，用于自动生成标签
):
    """
    ✅ 增强版 Krum 防御方法
    - 支持自定义选中比例 select_ratio
    - 自动映射标签到当前参与客户端
    - 正确计算 TPR / TNR / PRC
    """

    n = len(participating_clients)
    if n == 0:
        raise ValueError("No participating clients this round.")

    # --- Step 1️⃣ 扁平化每个客户端的参数向量 ---
    client_tensors = []
    for client in participating_clients:
        update = []
        for name, p in client.model.state_dict().items():
            update.append(p.detach().cpu().flatten())
        client_tensors.append(torch.cat(update))
    client_tensors = torch.stack(client_tensors).to(device)

    # --- Step 2️⃣ 计算两两欧氏距离矩阵 ---
    distances = torch.cdist(client_tensors, client_tensors, p=2).pow(2)

    # --- Step 3️⃣ 计算每个客户端的 Krum 分数 ---
    scores = []
    for i in range(n):
        nearest_distances, _ = torch.sort(distances[i])
        score = torch.sum(nearest_distances[1: n - f - 1])
        scores.append(score.item())
    scores = np.array(scores)

    # --- Step 4️⃣ 选出得分最低的前 m 个客户端 ---
    m = int(max(1, n * select_ratio))
    m = min(m, n)
    selected_indices = np.argsort(scores)[:m]

    # --- Step 5️⃣ 聚合选中客户端的模型参数 ---
    new_state_dict = deepcopy(server_model.state_dict())
    with torch.no_grad():
        for key in new_state_dict.keys():
            selected_tensors = [participating_clients[i].model.state_dict()[key].float() for i in selected_indices]
            new_state_dict[key] = torch.mean(torch.stack(selected_tensors), dim=0)

    # --- Step 6: 映射并构造全局真实标签（如果没传 overall_label，则按 id>=total_clients/2 为恶意） ---
    if overall_label is None:
        if total_clients is not None:
            overall_label = [False] * total_clients
            # 你实验中 id 15-29 是恶意：把后半标为 True (malicious)
            # 若想通用化，使用 total_clients//2 分界
            for idx in range(total_clients // 2, total_clients):
                overall_label[idx] = True
        else:
            # fallback：全部认为良性（不建议）
            overall_label = [False] * n

    # 全局标签数组（0/1），1 表示真实为 malicious
    y_true_full = np.array([1 if flag else 0 for flag in overall_label])

    # --- Step 7: 计算局部预测并转为全局预测 ---
    # selected_indices 是局部索引（0..n-1）——已在前面计算
    selected_global_ids = [participating_clients[i].id for i in selected_indices]

    # 现在定义 predicted_malicious: 未被选中 => 1 (预测为 malicious)
    # 构造与全局长度一致的预测数组 y_pred_full
    y_pred_full = np.zeros_like(y_true_full, dtype=int)
    # 标记所有参与客户端为 默认 未选中（predicted_malicious = 1），然后把选中的置为 0
    # （先把参与者全部设为 1，再对 selected_global_ids 置 0）
    for c in participating_clients:
        y_pred_full[c.id] = 1
    # 选中的认为 benign -> predicted_malicious = 0
    for gid in selected_global_ids:
        y_pred_full[gid] = 0

    # --- Step 8: 现在计算混淆矩阵（以 malicious=1 为正类） ---
    TP = int(np.sum((y_pred_full == 1) & (y_true_full == 1)))  # 正确识别为恶意
    FP = int(np.sum((y_pred_full == 1) & (y_true_full == 0)))  # 错误判为恶意的良性
    TN = int(np.sum((y_pred_full == 0) & (y_true_full == 0)))  # 正确识别为良性
    FN = int(np.sum((y_pred_full == 0) & (y_true_full == 1)))  # 漏判的恶意（被误认为良性）

    TPR = TP / (TP + FN + 1e-8)  # recall for malicious
    TNR = TN / (TN + FP + 1e-8)  # true negative rate
    PRC = TP / (TP + FP + 1e-8)  # precision for malicious

    metrics = {"TP": TP, "FP": FP, "TN": TN, "FN": FN,
               "TPR": TPR, "TNR": TNR, "PRC": PRC}

    # 打印用全局 id 列表，避免混淆
    print(f"[Krum] selected global clients -> {selected_global_ids}")
    print(f"[Krum-Detect] TP={TP} FP={FP} TN={TN} FN={FN} | TPR={TPR:.3f} TNR={TNR:.3f} PRC={PRC:.3f}")

    return new_state_dict, selected_global_ids, metrics
