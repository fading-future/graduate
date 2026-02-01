import tifffile
import numpy as np

CONFIG = {
    'input_npy_path': r"E:\aligned_Training_Data\6-6-20 全部_z640_y530_x449.npy",
    'output_tiff_path': 'output2.tiff'
}

data = np.load(CONFIG['input_npy_path'])
tifffile.imwrite(CONFIG['output_tiff_path'], data, imagej=True) # 使用 ImageJ 兼容模式可保存 resolution

