#%%
import os
import glob
import re

# 修改为你的 6-6-20 文件夹路径
folder_path = r"/chendou_space/data/core_ctimg_data/6-6-20 全部"

#%%
def check_filename_sorting():
    files = glob.glob(os.path.join(folder_path, "*.tif"))
    
    if not files:
        print("❌ 文件夹为空或路径错误")
        return

    # 1. 模拟你现在的错误排序逻辑
    def current_sort_key(x):
        match = re.search(r'modif(\d+)', x)
        return int(match.group(1)) if match else 0
    
    bad_sort = sorted(files, key=current_sort_key)
    
    print("--- 😱 你现在的排序 (前10张) ---")
    for f in bad_sort[:10]:
        print(os.path.basename(f))
        
    print("\n--- 分析 ---")
    first_file = os.path.basename(bad_sort[0])
    if "modif" not in first_file:
        print(f"❌ 破案了！文件名 '{first_file}' 不包含 'modif'！")
        print("   正则 re.search(r'modif(\d+)') 匹配失败，导致乱序。")
    else:
        # 如果包含 modif 但还是乱序，可能是数字提取逻辑有问题
        pass

    # 2. 尝试通用数字排序 (自然排序)
    # 这种逻辑会提取文件名里的任意数字进行排序，不管有没有 modif
    def smart_sort_key(x):
        # 提取文件名中的所有数字，取最后一个作为排序依据
        numbers = re.findall(r'\d+', os.path.basename(x))
        return int(numbers[-1]) if numbers else 0
        
    good_sort = sorted(files, key=smart_sort_key)
    
    print("\n--- ✅ 建议的正确排序 (前10张) ---")
    for f in good_sort[:10]:
        print(os.path.basename(f))

if __name__ == "__main__":
    check_filename_sorting()
# %%
