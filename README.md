

毕业论文涉及到的所有代码、图件和说明信息都在这里

🎓【基于潜在扩散模型的岩心一致性条件扩展生成】

> *核心参考文献：`"E:\chendou\Ren 等 - 2024 - Constrained Transformer-Based Porous Media Generation to Spatial Distribution of Rock Properties.pdf"`*





# `paper_code` 代码使用指南

## 1. 基本信息介绍

``````
E:\chendou\paper_code> tree /F
E:.
├─stage01_data_process_code
│  │  README.md
│  └─src
│      ├─step1
│      │      ctimg2npy_v1.py
│      │      ctimg2npy_v2.py   
│      ├─step2
│      │      calc_porosity.py  
│      └─visualization
│          │  step1_visual_v1.py
│          │  step1_visual_v2.py
│          │  step2_visual.py
│          │  
│          └─img_data
│                  Step1_Fig_Data_Construction.png
│                  Step1_Thesis_Figure_MethodNoise.png
│                  Step2_Figure_1_Distribution_Wide.png
│                  Step2_Figure_2_Processing_Flow_Reordered.png
├─stage02_vqvae_code
│  │  README.md
│  ├─exp_results
│  │  └─exp01
│  │      ├─log
│  │      │      training_curve_paper.png
│  │      │      viz_epoch_110.png
│  │      └─model
│  │              vqvae_epoch_110.pth
│  ├─src
│  │      config.py
│  │      dataset_rev.py
│  │      model_vqvae.py
│  │      train_vqvae.py
│  └─utils
│          get_root_path.py
├─stage03_latent_ddpm_code
│  ├─exp_results
│  │  └─exp01
│  │      ├─log
│  │      └─model
│  ├─src
│  └─utils
├─stage04_result_analyze_code
└─utils
        check_visual_npy_stats.py
        prepare_latent_npy.py
        visual_rev_npy_threeFacets.py
``````



`paper_code` 主要分为以下几个模块:

1. `stage01_data_process_code` 数据处理部分
2. `stage02_KLvae_single_code_v2` 基于潜在扩散模型的第一阶段: KLVAE 模型
3. `stage07_latent_ldm_code` 基于潜在扩散模型的第二阶段: LDM 模型
4. `stage04_result_analyze_code` 结果分析部分. 论文中涉及到的绘图脚本基本都在这里

其余部分为尝试过的其他策略, 效果不理想. 这里做简要的介绍:

- 尝试一: 基于像素空间的扩散模型的尝试. 在`stage03_pixelddpm_code` 中直接在像素空间做三面掩码策略训练条件扩散模型. 效果一般. 计算量大, 孔隙率条件没有很好的学习到. 弃用  
- 尝试二: 基于潜空间扩散模型的尝试, 使用VQVAE 或者KLVAE 作为第一阶段模型, 在潜在空间做三面掩码策略训练LDM, 但是存在孔隙不联通, 生成的孔隙散碎的问题. 具体来说:在`stage02_vqvae_code` 中尝试使用VQVAE 作为潜在扩散模型的第一阶段, 使用`stage03_latent_ddpm_code` 条件ddpm 模型 作为潜在扩散模型的第二阶段.
- 尝试三: 基于潜在扩散模型的尝试, 第一阶段与尝试二中保持一致, 但是在训练第二阶段的LDM 的时候, 针对数据集进行了改进, 具体见代码, 但是效果依旧很差

以上的尝试均是在灰度NPY 数据上进行的. 并且针对灰度数据的分布, 掩码策略, VQVAE 死码问题, KLVAE 模糊问题等, 做了各种优化, 效果均一般. 因此后续参考 Ren 等人中使用VQVAE + Transformer 的自回归生成方式对代码进行了重构和优化, 确定了目前的:

- 数据处理: 使用二值化之后的数据训练两阶段模型
- 两阶段模型: 基于LDM 的自回归生成
  - 第一阶段使用KLVAE 提到VQVAE 压缩数据, 因为KLVAE 产生的连续潜在空间更适合LDM 的训练. 
  - 第二阶段使用LDM 做自回归生成, 摒弃了三面掩码策略的训练方式



## 2. 如何训练运行

整个项目存放在`chendou/paper_code` 目录，通过`uv` 管理项目的python 版本、包版本、包依赖关系以及虚拟环境*（学习`uv`。移步到 >>> Appendix.1）*

代码推荐运行方式：

1. 打开终端，进入到某个阶段的代码的根目录，比如我要运行`stage02_vqvae_code` 中的程序，那就需要控制台显示下面的路径`(chendou) PS E:\chendou\paper_code\stage02_vqvae_code> `  
2. 在终端中，使用`python -m src.train` 命令来运行对应的python 代码。直接点击`vscode` 右上角的运行按钮可能会报错。

> （python 中的不同启动方式有不同的作用. 想要搞懂到底有什么区别的话。*移步到 >>> Appendix.2*）
>
> **不同的程序启动方式，直接运行文件 vs 使用 `-m` 运行**
>
> 1. 直接运行文件：相当于在命令行中任意目录，直接使用`python 脚本的绝对路径or相对路径.py` 
>    1. 普通的小脚本的常用运行方式，但是如果涉及到以及比较大的项目，会有坑。
>    2. 当涉及到自己写的脚本之间需要互相导入的时候，直接运行文件会把**脚本所在的目录**（即 `.../stage02_vqvae_code/src`）加入到 `sys.path` 的第一个位置。结果就是脚本可以找到同级目录`src` 下的其他脚本，**但是**，脚本找不到上一层的 `utils` 目录！因为 `src` 是根，当前脚本它看不到外面。
> 2. 使用`-m` 运行：只能在当前项目的根目录，使用`python 脚本相对于项目根目录的路径` 中间使用`.` 隔开，并且结尾不加`.py`



### 步骤一: 准备数据

> 1. 原始CT 数据信息: 移步到 >>> Appendix.6
> 2. 基本上数据处理的代码都在`stage01_data_process_code` 中

*⚠️数据处理流程: `原始CT 数据 ---> NLM 去噪之后的CT 数据 ---> 阈值分割之后的二值CT 数据 ---> 滑动窗口采样获取的NPY 数据 ---> KLVAE 压缩之后的Latent NPY 数据 ---> 训练LDM 所需要的Pairs NPY 数据`* 

具体介绍:

1️⃣ NLM 去噪之后的 CT 数据:

- 获取方式: 运行`E:\chendou\paper_code\stage01_data_process_code\src\step_additional\grayimg2peak2nlmimg.py` 
- 输入数据: 原始的CT 数据
- 输出路径: `D:\多尺度岩心数据集\Lastest_Preprocess\Gray_Preprocessed_Slices`

2️⃣ 阈值分割之后的二值CT 数据:

- 获取方式: 运行`E:\chendou\paper_code\stage01_data_process_code\src\step_additional\preprocessed2binary.py`
- 输入数据: NLM 去噪之后的CT 数据
- 输出路径: `D:\多尺度岩心数据集\Lastest_Preprocess\Binary_Preprocessed_Slices`

3️⃣ 滑动窗口采样获取的NPY 数据:

- 获取方式: 运行`E:\chendou\paper_code\stage01_data_process_code\src\step_additional\binaryct2npy.py`
- 输入数据: 阈值分割之后的二值CT 数据
- 输出路径: `D:\多尺度岩心数据集\LDM_Data\Raw_NPY\w192_s64`

4️⃣ KLVAE 压缩之后的Latent NPY 数据:

- 获取方式: 运行`E:\chendou\paper_code\stage02_KLvae_single_code_v2\inference.py`
- 输入数据:滑动窗口采样获取的NPY 数据
- 输出路径: `D:\多尺度岩心数据集\LDM_Data\Latent_NPY\w192_s64`

5️⃣ 训练LDM 所需要的Pairs NPY 数据之一的Phi Maps NPY 数据:

- 获取方式: 运行`E:\chendou\paper_code\stage07_latent_ldm_code\src\preprocess_phi.py`
- 输入数据: 滑动窗口采样获取的NPY 数据
- 输出路径: `D:\多尺度岩心数据集\LDM_Data\Phi_Maps_NPY\w192_s64` 



### 步骤二: 训练KLVAE

``````bash
# 方式一: 重新开启一个新的训练
(chendou) PS E:\chendou\paper_code\stage02_KLvae_single_code_v2> python -m train

# 方式二: 从某个模型参数开始继续训练
(chendou) PS E:\chendou\paper_code\stage02_KLvae_single_code_v2> python -m train --config config/train_config.yaml --resume ./experiments/exp05_cube_structure_v2/ckpt_epoch_11.pt
``````



### 步骤三: 训练LDM

```````bash
# 开启一个新的训练 or 从最新的模型参数开始继续训练
(chendou) PS E:\chendou\paper_code\stage07_latent_ldm_code> python -m src.train
```````







# Appendix

## 1. `uv` 包管理工具









## 2. 一文说明白python 自定义模块的导入机制

> 学习是个螺旋上升的过程，现在似懂非懂也没关系，相同的问题总会反复遇见，不必急于求成。**只需多思勤练，待到熟能生巧，领悟自在其中。**

当前小节尝试回答的问题：

1. python 脚本中到底怎么导入自定义的包（自己写的其他的python 脚本）呢？不同的导入方式有什么区别，会带来什么影响呢？
2. python 脚本到底怎么运行呢？直接运行和使用`-m` 运行有什么区别，会带来什么影响呢？
3. python 脚本中到底怎么获取路径信息？工作目录和绝对路径有什么区别呢？
4. 上面三个问题是怎么耦合在一起，相互影响的呢？
5. 其他编程语言`Java, Golang` 怎么做的？为什么没有这些问题？



## 3. 云算力平台使用指南




## 4. Ubuntu 镜像使用指南



## 5. 远程连接Windows 或者Ubuntu





## 6. 原始数据介绍

```
目录					深度(m)			总长度(cm)			直径(mm)		图片数量
D:\多尺度岩心数据集	
├─6-6-9				26.5				55.03			100				11701
├─6-6-12			35.5				37				100				7612	
├─6-6-15			46					67				100				12726
├─6-6-18			57.5				37.5			75				8173
├─6-6-20 全部			70					31.5			50				7921
├─6-6-21			74					39.6			50				7892
├─6-6-22			83					29.5			50				6237
├─6-6-23			90					37				50				9219
└─6-6-24			97					13				50				3060
```

1. 以上数据的分辨率均为 56 $\mu m$ , 图片的尺寸均为 $1900 \times 1900$
2. 图片的边缘存在伪影
3. 整个岩心柱不是标准的圆柱体，存在破碎和残缺
4. 深度 26.5-97m, 图片总计数量约7w 张

# Error(踩坑指南)
## 1. `import torch` 导入出错
现象：在uv 虚拟环境中`import torch` 导入出错，但是在系统的python 环境中`import torch` 导入成功
错误信息：
原因：镜像环境缺少驱动文件导致的错误

## 2. Windows 环境下利用accelerate 库做多卡训练
现象：使用多卡训练，频繁出现环境变量设置，网络通信问题
错误信息：
原因：Windows 中对于accelerate 多卡训练库的支持不好。推荐Windows 中直接使用原生的多卡训练库



