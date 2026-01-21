```shell
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
│          └─img_data
│                  Step1_Fig_Data_Construction.png
│                  Step1_Thesis_Figure_MethodNoise.png
│                  Step2_Figure_1_Distribution_Wide.png
│                  Step2_Figure_2_Processing_Flow_Reordered.png
├─stage02_vqvae_code
│  │  README.md
│  └─src
├─stage03_latent_ddpm_code
├─stage04_result_analyze_code
└─utils
        visual_rev_npy_threeFacets.py
```

# 一、数据基本信息
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

小结
1. 深度 26.5-97m
2. 图片数量约7w 张
3. 以上数据的分辨率均为 56 $\mu m$
4. 图片的尺寸均为 $1900 \times 1900$
5. 图片的边缘存在伪影
6. 整个岩心柱不是标准的圆柱体，存在破碎和残缺
```

# 二、数据处理流程
step1 --> step2

**数据流转：**
|**阶段**|**数据形状 (Batch, C, D, H, W)**|**数据类型**|**数值范围示例**|
|---|---|---|---|
|**原始文件**|`(256, 256, 256)` _无Batch/Channel_|`uint16`|`11698` ~ `38965`|
|**网络输入**|`(1, 1, 256, 256, 256)`|`float32`|`-1.0` ~ `1.0`|
|**Encoder输出**（即latent 立方体数据）|`(1, 64, 64, 64, 64)`|`float32`|约 `-2.0` ~ `2.0` (由码本决定)|
|**Decoder输出**|`(1, 1, 256, 256, 256)`|`float32`|`-1.0` ~ `1.0`（由 Decoder 的**最后一层激活函数Tanh**决定）|

