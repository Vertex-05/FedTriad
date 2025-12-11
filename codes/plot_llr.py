import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_llr(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = np.load(file_path, allow_pickle=True)
    if 'sprt_llr_history' not in data:
        print("Error: 'sprt_llr_history' key not found in .npz file.")
        print("Please ensure you have modified run_ours.py to log LLR values.")
        return

    # shape: (Rounds, Clients)
    llrs = data['sprt_llr_history']
    rounds = np.arange(1, llrs.shape[0] + 1)
    n_clients = llrs.shape[1]

    # 获取攻击比例，区分良性/恶意（假设后半部分是恶意）
    hp = data['hyperparameters'].item()
    att_rate = hp.get('attack_rate', 0.5)
    n_malicious = int(n_clients * att_rate)
    n_benign = n_clients - n_malicious
    
    # 阈值
    alpha = hp.get('sprt_alpha', 0.01)
    beta = hp.get('sprt_beta', 0.05)
    threshold = np.log((1 - beta) / alpha)

    plt.figure(figsize=(10, 6))
    
    # 画良性 (前 n_benign 个) - 蓝色
    for i in range(n_benign):
        label = 'Benign' if i == 0 else None
        plt.plot(rounds, llrs[:, i], c='blue', alpha=0.3, label=label)

    # 画恶意 (后 n_malicious 个) - 红色
    for i in range(n_benign, n_clients):
        label = 'Malicious' if i == n_benign else None
        plt.plot(rounds, llrs[:, i], c='red', alpha=0.6, label=label)

    plt.axhline(y=threshold, c='black', linestyle='--', label='Threshold (LogA)')
    plt.xlabel('Communication Rounds')
    plt.ylabel('SMI (Cumulative LLR)')
    plt.title(f'SMI Dynamics (Attack: {hp.get("attack_method", "Unknown")})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = file_path.replace('.npz', '_llr.png')
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, required=True)
    args = parser.parse_args()
    plot_llr(args.log_file)