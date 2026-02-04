import os
import glob
import numpy as np
import tifffile
import re

def fill_missing_images(folder_path):
    # 1. 定义文件名的各个部分
    # 根据你的描述：FdkRecon-ushort-1900x1900x9624.modif0000.tif
    file_prefix = "FdkRecon-ushort-1900x1900x9624.modif"
    file_suffix = ".tif"
    
    # 目标范围：0000 到 3058
    start_index = 0
    end_index = 3058
    
    # 2. 获取该文件夹下所有符合规则的文件
    search_pattern = os.path.join(folder_path, f"{file_prefix}*{file_suffix}")
    existing_files = glob.glob(search_pattern)
    
    if not existing_files:
        print("错误：在指定路径下没有找到符合命名规则的文件。请检查路径或文件名前缀。")
        return

    # 3. 找出存在的序号
    existing_indices = set()
    # 用来作为模板的文件路径（取第一个找到的文件）
    template_file_path = existing_files[0] 
    
    print(f"正在扫描文件夹: {folder_path}")
    print(f"使用模板文件读取参数: {os.path.basename(template_file_path)}")

    for file_path in existing_files:
        filename = os.path.basename(file_path)
        # 提取中间的数字部分
        try:
            # 移除前缀和后缀，剩下应该是数字
            num_str = filename.replace(file_prefix, "").replace(file_suffix, "")
            idx = int(num_str)
            existing_indices.add(idx)
        except ValueError:
            continue # 如果文件名不符合预期数字格式，跳过

    # 4. 读取模板文件的属性 (尺寸, 数据类型)
    try:
        # 读取原始图像数据
        template_data = tifffile.imread(template_file_path)
        height, width = template_data.shape
        dtype = template_data.dtype
        
        print(f"检测到图像尺寸: {width}x{height}")
        print(f"检测到数据类型: {dtype} (应为 uint16/ushort)")
        
    except Exception as e:
        print(f"读取模板文件失败: {e}")
        return

    # 5. 创建全黑图像数据 (数值全为0，且保持原有数据类型)
    black_data = np.zeros_like(template_data)

    # 6. 循环检查缺失的序号并补全
    missing_count = 0
    print("-" * 30)
    
    for i in range(start_index, end_index + 1):
        if i not in existing_indices:
            # 构造缺失文件的完整路径
            # :04d 表示补零至4位，例如 5 -> 0005
            new_filename = f"{file_prefix}{i:04d}{file_suffix}"
            save_path = os.path.join(folder_path, new_filename)
            
            # 保存全黑图片
            # photometric='minisblack' 显式声明黑白模式，虽非必须但推荐
            tifffile.imwrite(save_path, black_data, photometric='minisblack')
            
            print(f"已补全: {new_filename}")
            missing_count += 1

    print("-" * 30)
    if missing_count > 0:
        print(f"任务完成！共补全了 {missing_count} 张缺失图片。")
    else:
        print("检查完毕，序列是完整的，无需补全。")

if __name__ == "__main__":
    # --- 配置区域 ---
    # 将下面的路径改为你图片所在的实际文件夹路径
    # 如果脚本就在图片文件夹里，可以使用 "."
    target_directory = r"D:\多尺度岩心数据集\6-6-24_Manual_Threshold" 
    
    # 也可以直接使用当前脚本所在目录
    # target_directory = os.path.dirname(os.path.abspath(__file__))

    fill_missing_images(target_directory)