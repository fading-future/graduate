

🎓毕业论文涉及到的所有代码和图件和说明信息都在这里

【基于潜在扩散模型的岩心一致性条件扩展生成】



分为四个阶段来完成：

1. `stage01_data_process_code` 数据处理部分
2. `stage02_vqvae_code` `vqvae 模型`部分
3. `stage03_latent_ddpm_code` `条件ddpm 模型`部分
4. `stage04_result_analyze_code` 结果分析部分

整个项目存放在`chendou` 目录，通过`uv` 管理项目的python 版本、包版本、包依赖关系以及虚拟环境（不会`uv` 的话快去学习一下，好用，童叟无欺）。

*（学习`uv`。移步到 >>> Appendix.1）*

`stage02_vqvae_code` 和 `stage03_latent_ddpm_code` 想要复现的话，需要比较高的算力。大概率组内现在最好的显卡还是A6000，大概率你也需要租用算力。

*（算力租用指南。移步到 >>> Appendix.3）*



项目推荐启动方式：

1. 打开终端，进入到某个阶段的代码的根目录，比如我要运行`stage02_vqvae_code` 中的程序，那就需要控制台显示下面的路径`(chendou) PS E:\chendou\paper_code\stage02_vqvae_code> `  
2. 在终端中，使用`python -m src.train` 命令来运行对应的python 代码。直接点击`vscode` 右上角的运行按钮会报错。

> （想省事就直接使用`-m` ，因为我的代码就是按照`-m` 运行不出错的逻辑写的）
>
> （想要彻底搞懂到底有什么区别的话。*移步到 >>> Appendix.2*）
>
> **不同的程序启动方式，直接运行文件 vs 使用 `-m` 运行**
>
> 1. 直接运行文件：相当于在命令行中任意目录，直接使用`python 脚本的绝对路径or相对路径.py` 
>    1. 普通的小脚本的常用运行方式，但是如果涉及到以及比较大的项目，会有坑。
>    2. 当涉及到自己写的脚本之间需要互相导入的时候，直接运行文件会把**脚本所在的目录**（即 `.../stage02_vqvae_code/src`）加入到 `sys.path` 的第一个位置。结果就是脚本可以找到同级目录`src` 下的其他脚本，**但是**，脚本找不到上一层的 `utils` 目录！因为 `src` 是根，当前脚本它看不到外面。
> 2. 使用`-m` 运行：只能在当前项目的根目录，使用`python 脚本相对于项目根目录的路径` 中间使用`.` 隔开，并且结尾不加`.py`





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





```
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

```



## 3. 云算力平台使用指南




## 4. Ubuntu 镜像使用指南



## 5. 远程连接Windows 或者Ubuntu



# Error(踩坑指南)
## 1. `import torch` 导入出错
现象：在uv 虚拟环境中`import torch` 导入出错，但是在系统的python 环境中`import torch` 导入成功
错误信息：
原因：镜像环境缺少驱动文件导致的错误

## 2. Windows 环境下利用accelerate 库做多卡训练
现象：使用多卡训练，频繁出现环境变量设置，网络通信问题
错误信息：
原因：Windows 中对于accelerate 多卡训练库的支持不好。推荐Windows 中直接使用原生的多卡训练库



