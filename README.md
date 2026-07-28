TongueAI

基于深度学习的中医舌象智能分析系统。

TongueAI 通过 YOLO-seg 舌体自动分割模型与EfficientNet-B3 多任务分类模型相结合，实现从原始舌象图片到结构化舌象特征分析的自动化流程。

项目主要用于中医舌诊场景下的舌体定位、舌象特征识别以及辅助智能诊断研究。

项目简介

传统中医舌诊高度依赖医生经验，存在主观性强、标准不统一、效率较低等问题。

TongueAI 使用计算机视觉与深度学习技术，将舌象分析流程自动化：

1. 自动定位舌体区域
2. 去除复杂背景干扰
3. 提取舌象视觉特征
4. 多任务学习同时预测多个舌象属性

整体采用两阶段架构：

Image
  |
  |
YOLO-Seg
  |
  |
Tongue ROI
  |
  |
EfficientNet-B3
  |
  |
Multi-task Classification
  |
  |
Tongue Features

系统架构


1. 舌体分割模块

模型：

YOLO11-segv11

作用：

- 检测舌体位置
- 生成舌体mask
- 提取舌体ROI区域


输入：

原始舌象图片

输出：

tongue mask

tongue ROI image

示例：

dataset/
|
├── images
│     └── tongue_000001.jpg
|
└── labels
      └── tongue_000001.txt


YOLO-seg标签格式：

class x1 y1 x2 y2 ... xn yn

2. 舌象分类模块

模型：

EfficientNet-B3
(tf_efficientnet_b3_ns)


预训练权重：

tf_efficientnet_b3_ns.safetensors


Backbone：

EfficientNet-B3

作用：

提取舌象高级视觉特征。

输入：

Tongue ROI image

输出：

feature vector

多任务分类 Head

采用 Multi-task Learning 结构：


                 Feature Vector

                       |
        --------------------------------
        |       |        |       |      |
     舌色    舌苔色   舌苔厚度  裂纹   齿痕


当前任务：


|任务|类别|

|舌色|3分类|
|舌苔颜色|3分类|
|舌苔厚薄|2分类|
|裂纹|2分类|
|齿痕|2分类|

标签映射：

json
{
    "tongue_color": {
        "0": "舌质淡白",
        "1": "舌质淡红",
        "2": "舌质红绛"
    },

    "coat_color": {
        "0": "舌苔白",
        "1": "舌苔薄白",
        "2": "舌苔黄"
    },

    "coat_thickness": {
        "0": "苔薄",
        "1": "苔厚"
    },

    "crack": {
        "0": "无裂纹舌",
        "1": "裂纹舌"
    },

    "teeth_mark": {
        "0": "无齿痕舌",
        "1": "齿痕舌"
    }
}

环境配置

Python

Python >=3.10

GPU环境

推荐：

CUDA >=12
GPU >=12GB

测试环境：

NVIDIA RTX 5090
CUDA 13.0
PyTorch 2.x

安装依赖：

bash
pip install -r requirements.txt

数据准备

舌体分割数据


目录：
tongue_seg/dataset

├── images

└── labels

其中：

- images：舌象原图
- labels：YOLO-seg polygon标签

分类数据

分类数据采用Excel管理：

dataset/

├── train.xlsx
├── val.xlsx
└── test.xlsx

字段：

|字段|说明|

|image|图片路径|
|tongue_color|舌色标签|
|coat_color|舌苔颜色|
|coat_thickness|舌苔厚度|
|crack|裂纹|
|teeth_mark|齿痕|

模型训练

1. 训练YOLO-seg

进入：
bash
cd tongue_seg

训练：
bash
python train.py

训练完成：
weights/best.pt
用于舌体自动分割。

2. 训练EfficientNet多任务模型

进入：
bash
cd tongue_classify

训练：
bash
python train.py

输出：
weights/best.pth

推理流程
完整推理流程：
input.jpg

      |
      v

YOLO-seg

      |
      v

tongue ROI

      |
      v

EfficientNet-B3

      |
      v

multi-task heads

      |
      v

tongue analysis result

运行预测：
bash
python predict.py \
--image demo.jpg

输出示例：
json
{
    "tongue_color":"舌质淡红",

    "coat_color":"舌苔白",

    "coat_thickness":"苔薄",

    "crack":"无裂纹舌",

    "teeth_mark":"齿痕舌"
}

模型设计说明

Backbone
采用：
EfficientNet-B3

优势：
- 参数量适中
- 特征提取能力强
- 适合医学小样本视觉任务

Head设计

传统方式：

一个模型
|
一个分类任务

本项目：
一个Backbone

       |

多个任务Head

优势：

- 共享舌象视觉特征
- 减少模型数量
- 提升任务之间关联学习能力

Loss设计
采用：
Weighted Cross Entropy
+
Focal Loss

针对医学数据类别不均衡问题：
- 增加少数类别关注
- 降低多数类别影响

Loss权重：
yaml
loss_weights:

  tongue_color: 1.0

  coat_color: 1.0

  coat_thickness: 1.0

  crack: 1.5

  teeth_mark: 1.5

当前模型能力
支持：

✅ 舌体自动定位

✅ 舌体mask生成

✅ ROI自动提取

✅ 舌色识别

✅ 舌苔颜色识别

✅ 舌苔厚薄识别

✅ 裂纹识别

✅ 齿痕识别

Future Work
计划：

- [ ] 引入SAM/TongueSAM提升分割精度
- [ ] 增加舌形分类任务
- [ ] 增加舌态分析
- [ ] 与中医知识库RAG系统融合
- [ ] 构建完整AI辅助舌诊平台

License

This project is for academic research and medical AI exploration only.
