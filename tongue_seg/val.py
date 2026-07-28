# val.py

from ultralytics import YOLO


# =============================
# 配置
# =============================

MODEL_PATH = "/root/project/tongue_seg/models/weights/small.pt"

DATA_CONFIG = "/root/project/tongue_seg/configs/dataset.yaml"

DEVICE = 0



def main():


    # 加载模型
    model = YOLO(MODEL_PATH)


    # 验证
    metrics = model.val(

        data=DATA_CONFIG,

        imgsz=1024,

        batch=16,

        device=DEVICE,

        workers=8,

        save_json=True

    )


    # 输出结果

    print("\n========== Validation Result ==========")


    print(
        "Box mAP50:",
        metrics.box.map50
    )


    print(
        "Box mAP50-95:",
        metrics.box.map
    )


    print(
        "Mask mAP50:",
        metrics.seg.map50
    )


    print(
        "Mask mAP50-95:",
        metrics.seg.map
    )



if __name__ == "__main__":

    main()