# ======================================================
# DP_defense.py (Device-safe version)
# Differential Privacy Defense module for Federated Learning
# ======================================================

import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal


# ======================================================
# 1. ClipOptimizer: 用于客户端本地梯度裁剪
# ======================================================
class ClipOptimizer(object):
    def __init__(self, base_optimizer, device, clip_norm=1.0, clip_freq=1):
        self.base_optimizer = base_optimizer
        self.device = device
        self.clip_norm = clip_norm
        self.clip_freq = clip_freq
        self.counter = 0

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def step(self, global_model_state, last_iter=False):
        self.base_optimizer.step()
        self.counter += 1
        if self.counter % self.clip_freq != 0 and not last_iter:
            return

        ordered_state_keys = list(global_model_state.keys())
        local_params = self.base_optimizer.param_groups[0]["params"]
        delta_params = [
            torch.sub(local_params[idx], global_model_state[ordered_state_keys[idx]].to(self.device))
            for idx in range(len(local_params))
        ]

        per_param_delta_norm = torch.stack([torch.norm(p) for p in delta_params])
        delta_norm = torch.norm(per_param_delta_norm)
        clip_factor = (self.clip_norm / (delta_norm + 1e-6)).clamp(max=1.0)

        for idx, param in enumerate(local_params):
            clipped_update = global_model_state[ordered_state_keys[idx]].to(self.device) + delta_params[idx] * clip_factor
            param.data = clipped_update


# ======================================================
# 2. FedAvg + Differential Privacy 聚合逻辑
# ======================================================
def FedAvg(models, previous_global_model=None, dp_params=None):
    """
    Standard FedAvg aggregation + optional DP clipping validation
    """
    # 自动检测设备
    device = next(models[0].parameters()).device

    # 所有 state dict 转到同一设备
    state_dicts = [{k: v.to(device) for k, v in model.state_dict().items()} for model in models]
    state_dict = {k: torch.zeros_like(v, device=device) for k, v in state_dicts[0].items()}

    for key in state_dict:
        state_dict[key] = sum([sd[key] for sd in state_dicts]) / len(models)

    new_model = models[0]
    new_model.load_state_dict(state_dict)

    # DP 参数校验与验证
    if dp_params is not None and previous_global_model is not None:
        prev_state = {k: v.to(device) for k, v in previous_global_model.state_dict().items()}
        for idx, nds in enumerate(state_dicts):
            delta_norms = torch.stack([torch.norm(nds[k] - prev_state[k]) for k in nds])
            total_norm = torch.norm(delta_norms)
            if total_norm > dp_params["clip_norm"]:
                print(f"[DP Warning] Client {idx} exceeded clip norm {dp_params['clip_norm']:.3f}")

    return new_model


# ======================================================
# 3. 向聚合后的模型添加噪声
# ======================================================
def add_noise_on_aggregated_parameters(model, num_clients, dp_params):
    """
    Adds Gaussian noise on aggregated model parameters after FedAvg
    """
    device = next(model.parameters()).device
    state_dict = model.state_dict()
    normal_distribution = Normal(
        loc=torch.tensor(0.0, device=device),
        scale=torch.tensor(dp_params["noise_multiplier"] * dp_params["clip_norm"], device=device)
    )

    with torch.no_grad():
        for name, tensor in state_dict.items():
            noise = normal_distribution.sample(tensor.shape)
            noise = noise / (num_clients * dp_params["sample_rate"])
            tensor.add_(noise)
        model.load_state_dict(state_dict)
    return model


# ======================================================
# 4. DP 参数合法性检查
# ======================================================
def validate_dp_params(dp_params):
    required_keys = ["clip_norm", "noise_multiplier", "delta", "sample_rate", "clip_frequency"]
    for key in required_keys:
        if key not in dp_params:
            raise ValueError(f"[DP Error] Missing key '{key}' in dp_params.")
    return True


# ======================================================
# 5. 核心接口：run_ours.py 调用的入口函数
# ======================================================
def dp_aggregate(participating_clients, server, dp_params):
    """
    Perform Differential Privacy Aggregation (FedAvg + Noise)
    Args:
        participating_clients: 当前轮参与的客户端列表
        server: Server 对象（包含全局模型）
        dp_params: 差分隐私参数字典
    Returns:
        global_model: 聚合并加噪后的新全局模型
    """
    validate_dp_params(dp_params)

    # 检测服务器模型所在设备
    device = next(server.models[0].parameters()).device

    # 收集客户端模型并确保一致设备
    local_models = [client.model.to(device) for client in participating_clients]

    # FedAvg 聚合
    new_global_model = FedAvg(local_models, previous_global_model=server.models[0], dp_params=dp_params)

    # 在服务器端添加噪声
    num_clients = len(participating_clients)
    new_global_model = add_noise_on_aggregated_parameters(new_global_model, num_clients, dp_params)

    print(f"[DP] Aggregation done (device={device}, σ={dp_params['noise_multiplier']}, clip_norm={dp_params['clip_norm']})")

    # === 自动兼容 evaluate_ensemble() 返回字典键 ===
    eval_results = server.evaluate_ensemble()
    acc_key = None
    for k in eval_results.keys():
        if "acc" in k.lower():   # 匹配 acc / accuracy / test_acc 等
            acc_key = k
            break
    if acc_key is None:
        print("[DP Warning] No accuracy key found in eval_results. Available keys:", list(eval_results.keys()))
        acc_value = 0.0
    else:
        acc_value = eval_results[acc_key]

    print(f"[DP] Round Evaluation Accuracy ({acc_key}): {acc_value:.4f}")
    return new_global_model
