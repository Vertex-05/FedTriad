import torch
import numpy as np

def coordinate_wise_median(updates):
    """
    对每个参数坐标独立计算中位数（coordinate-wise median）
    updates: List[np.ndarray] or List[torch.Tensor]
    return: torch.Tensor (聚合后的向量, 在 CPU 上)
    """
    if isinstance(updates[0], torch.Tensor):
        updates = [u.detach().cpu().numpy() for u in updates]
    stacked = np.stack(updates, axis=0)  # [n_clients, d]
    median_update = np.median(stacked, axis=0)
    return torch.tensor(median_update, dtype=torch.float32)  # 默认CPU


def aggregate_clients_with_median(participating_clients, server):
    """
    Median 聚合主逻辑：
    - 收集客户端上传的参数或梯度
    - 对每个参数坐标取中位数
    - 更新全局模型参数
    """
    print(f"[Median Aggregation] Round start with {len(participating_clients)} clients")

    # 🔹 检测当前设备（CPU / GPU）
    device = next(server.models[0].parameters()).device

    # Step 1️⃣ 收集每个客户端的权重更新
    updates = []
    for client in participating_clients:
        local_params = torch.nn.utils.parameters_to_vector(client.model.parameters()).detach().cpu()
        global_params = torch.nn.utils.parameters_to_vector(server.models[0].parameters()).detach().cpu()
        update = (local_params - global_params).numpy()
        updates.append(update)

    # Step 2️⃣ 坐标中位数聚合（在CPU上计算）
    median_update = coordinate_wise_median(updates)

    # Step 3️⃣ 更新全局模型参数（转换回原设备）
    global_vector = torch.nn.utils.parameters_to_vector(server.models[0].parameters()).detach().to(device)
    new_state_vector = global_vector + median_update.to(device)

    # Step 4️⃣ 将更新后的参数载入模型
    torch.nn.utils.vector_to_parameters(new_state_vector, server.models[0].parameters())

    print(f"[Median Aggregation] Done. Model updated on device: {device}")
    return server.models[0].state_dict()

