from pathlib import Path
import sys

# # -----------------测试的时候需要导入config 配置文件读取路径信息--------------------------------------------
# # 动态将项目根目录添加到 sys.path
# # 获取当前文件所在目录的父目录，即项目根目录 (your_project/)
# # 然后将其添加到 Python 路径中
# current_dir = Path(__file__).resolve().parent
# project_root = current_dir.parent
# sys.path.append(str(project_root))
# # -------------------------------------------------------------------------------------------------------
# # 现在，Python 知道去 your_project/ 目录找文件了
# from config_loader.config import CONFIG

def get_root_path() -> Path:
    """
    获取项目根目录的绝对路径（src 或项目主目录的路径）。

    假设此函数（或调用它的脚本）位于项目的某个子目录中。
    它通过找到当前文件的位置，并向上一级或多级目录回溯来确定根目录。
    """
    # # 查找main.py 所在的目录作为项目根目录
    # current_dir = Path(__file__).resolve().parent
    # for parent in [current_dir] + list(current_dir.parents):
    #     if (parent / "main.py").exists():
    #         return parent
    # # 如果没有找到 main.py，则报错
    # raise FileNotFoundError("未找到项目根目录（缺少 main.py 文件）")

    # 方法二：假设项目根目录是当前文件的上两级目录
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent  # 假设项目根目录是当前文件的父目录
    return project_root

if __name__ == "__main__":
    root_path = get_root_path()
    print(f"项目根目录绝对路径：{root_path}")
    # print(f"配置文件中的数据集目录：{CONFIG['data_dir']}")