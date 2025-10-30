import numpy as np
import pprint
import os
import sys


def load_npz(filename):
    """加载并打印 npz 文件中的所有内容（包含 CrowdGuard & SPRT 记录解析）"""
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        sys.exit(1)

    data = np.load(filename, allow_pickle=True)

    # --- 打印概要 ---
    print("\n================= SUMMARY =================")
    for key in data.files:
        arr = data[key]
        print(f"{key:<30} | shape={arr.shape} | dtype={arr.dtype}")
    print("===========================================\n")

    # --- 遍历所有变量 ---
    for key, arr in data.items():
        print("=" * 60)
        print(f"Variable: {key}")
        print(f"Type    : {type(arr)}")

        # ==== 特殊处理 CrowdGuard + SPRT 数据 ====
        if key == "crowdguard_round_groups":
            print(f"Length  : {len(arr)}")
            print("Sample  :", arr[:1])
            print("\n============================================================")
            print("=== CrowdGuard & SPRT Summary per Round ===\n")

            flat_rounds = []
            for elem in arr:
                # 多层嵌套安全展开
                if isinstance(elem, dict):
                    flat_rounds.append(elem)
                elif isinstance(elem, np.ndarray):
                    for sub in elem:
                        if isinstance(sub, dict):
                            flat_rounds.append(sub)
                        elif isinstance(sub, np.ndarray):
                            for ssub in sub:
                                if isinstance(ssub, dict):
                                    flat_rounds.append(ssub)

            # --- 打印每轮 ---
            for i, entry in enumerate(flat_rounds):
                if not isinstance(entry, dict):
                    continue
                print(f"---- Round {entry.get('round', i + 1)} ----")
                for field in [
                    "soft_group",
                    "defer_group",
                    "hard_group",
                    "soft_final_malicious",
                    "defer_final_malicious",
                    "hard_malicious",
                    "removed_clients",
                ]:
                    print(f"{field:22}: {entry.get(field, [])}")
            print("\n============================================")
            continue  # crowdguard 已单独打印，跳过默认逻辑

        # ==== 普通对象 ====
        if arr.shape == ():
            try:
                item = arr.item()
                if isinstance(item, dict):
                    print(f"Keys    : {list(item.keys())}")
                    pprint.pprint(item, indent=1, width=120)
                else:
                    print("Value   :", item)
            except Exception:
                print("Value   :", arr)
        elif key == "participating_clients":
            print(f"Shape   : {arr.shape}")
            print(f"Dtype   : {arr.dtype}")
            for i, row in enumerate(arr):
                print(f"Round {i+1:03d}: {row.tolist()}")

        print()

    # 若没有 crowdguard_round_groups
    if "crowdguard_round_groups" not in data:
        print("\nNo 'crowdguard_round_groups' key found in this file.")
        print("If you recently modified run_ours.py to include this logging, "
              "make sure you re-ran training to produce a new .npz file.\n")

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_log.py /root/autodl-tmp/FedREDefense/results/532/logfiles/xp_4279443.npz")
        sys.exit(0)

    filename = sys.argv[1]
    load_npz(filename)
