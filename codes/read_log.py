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

        # ==== 1. 特殊处理 CrowdGuard + SPRT 数据 ====
        if key == "crowdguard_round_groups":
            print(f"Length  : {len(arr)}")
            if len(arr) > 0:
                print("Sample  :", arr[:1])
                print("\n============================================================")
                print("=== CrowdGuard & SPRT Summary per Round ===\n")
                
                # 兼容处理：有些版本可能是嵌套数组，有些是直接列表
                flat_rounds = []
                for elem in arr:
                    if isinstance(elem, dict):
                        flat_rounds.append(elem)
                    elif isinstance(elem, np.ndarray): # 处理可能的嵌套
                        if elem.shape == (): # 0-d array wrapping dict
                            flat_rounds.append(elem.item())
                        else:
                            for sub in elem:
                                if isinstance(sub, dict):
                                    flat_rounds.append(sub)
                
                # 打印每轮详情
                for i, entry in enumerate(flat_rounds):
                    if not isinstance(entry, dict): continue
                    print(f"---- Round {entry.get('round', i + 1)} ----")
                    for field in ["soft_group", "defer_group", "hard_group", 
                                  "soft_final_malicious", "defer_final_malicious", "hard_malicious", "removed_clients"]:
                        print(f"{field:22}: {entry.get(field, [])}")
                print("\n============================================")
            else:
                print("Empty Array")
            continue

        # ==== 2. 特殊处理 Participating Clients ====
        if key == "participating_clients":
            print(f"Shape   : {arr.shape}")
            print(f"Dtype   : {arr.dtype}")
            # 只打印前5轮和后5轮，避免刷屏
            total_rows = arr.shape[0]
            for i in range(total_rows):
                if total_rows > 20 and 5 < i < total_rows - 5:
                    if i == 6: print("   ... (omitting middle rounds) ...")
                    continue
                print(f"Round {i+1:03d}: {row_to_str(arr[i])}")
            print()
            continue

        # ==== 3. 处理标量或字典（shape为空） ====
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
        
        # ==== 4. [新增] 处理普通数组 (比如 accuracy, lr 等) ====
        else:
            # 如果数据量小，直接打印
            if arr.size < 50:
                print(f"Value   : {arr}")
            else:
                # 数据量大，打印统计信息或首尾
                print(f"Shape   : {arr.shape}")
                print(f"First 5 : {arr.flatten()[:5]}")
                print(f"Last 5  : {arr.flatten()[-5:]}")
                print(f"Mean    : {np.mean(arr):.4f}")
                print(f"Max     : {np.max(arr):.4f}")

        print()

    print("Done.")

def row_to_str(row):
    """辅助函数：将numpy行转为紧凑字符串"""
    return "[" + ", ".join(map(str, row.tolist())) + "]"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_log.py <path_to_npz_file>")
        sys.exit(0)

    filename = sys.argv[1]
    load_npz(filename)