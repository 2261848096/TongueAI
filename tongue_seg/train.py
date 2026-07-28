# ==========================================================
# YOLO11n-Seg 舌象分割训练脚本
# ==========================================================


from ultralytics import YOLO
import yaml
import os

# ==========================================================
# 项目根目录
# ==========================================================

PROJECT_DIR = "/root/project/tongue_seg"

# ==========================================================
# 配置文件路径
# ==========================================================


TRAIN_CONFIG = os.path.join(
    PROJECT_DIR,
    "configs/train.yaml"
)


DATA_CONFIG = os.path.join(
    PROJECT_DIR,
    "configs/dataset.yaml"
)


# ==========================================================
# 读取 train.yaml
# ==========================================================

with open(
    TRAIN_CONFIG,
    "r",
    encoding="utf-8"
) as f:

    cfg = yaml.safe_load(f)



# ==========================================================
# 加载YOLO分割模型
# ==========================================================

model_path = cfg["model"]


print("=" * 60)
print("加载模型:")
print(model_path)
print("=" * 60)



model = YOLO(model_path)



# ==========================================================
# 开始训练
# ==========================================================


results = model.train(

    # =============================
    # 数据集
    # =============================

    data=DATA_CONFIG,


    # =============================
    # 基础训练参数
    # =============================

    epochs=cfg["epochs"],

    imgsz=cfg["imgsz"],

    batch=cfg["batch"],

    seed=cfg["seed"],



    # =============================
    # 优化器
    # =============================

    optimizer=cfg["optimizer"],

    lr0=cfg["lr0"],

    lrf=cfg["lrf"],

    weight_decay=cfg["weight_decay"],



    # =============================
    # 训练控制
    # =============================

    patience=cfg["patience"],

    workers=cfg["workers"],

    amp=cfg["amp"],

    cache=cfg["cache"],



    # =============================
    # 学习率策略
    # =============================

    cos_lr=cfg["cos_lr"],



    # =============================
    # 正则化
    # =============================

    label_smoothing=cfg["label_smoothing"],


    # =============================
    # 图像增强
    # =============================

    hsv_h=cfg["hsv_h"],

    hsv_s=cfg["hsv_s"],

    hsv_v=cfg["hsv_v"],


    degrees=cfg["degrees"],

    translate=cfg["translate"],

    scale=cfg["scale"],

    fliplr=cfg["fliplr"],



    # =============================
    # Segmentation增强
    # =============================

    mosaic=cfg["mosaic"],

    mixup=cfg["mixup"],

    copy_paste=cfg["copy_paste"],



    # =============================
    # 保存配置
    # =============================

    project=os.path.join(
        PROJECT_DIR,
        "runs"
    ),

    name="yolo11s_seg_tongue",


    save=True,

    save_period=10,


    plots=True,


    # GPU
    device=0,

)



# ==========================================================
# 训练完成
# ==========================================================


print("\n" + "=" * 60)

print("训练完成!")

print("=" * 60)


print(
    "最佳模型路径:"
)

print(
    results.save_dir / "weights/best.pt"
)