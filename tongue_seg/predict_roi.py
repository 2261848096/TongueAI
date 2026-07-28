from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


# ==========================
# 参数
# ==========================

MODEL_PATH = (
    "/root/autodl-tmp/project/tongue_seg/models/weights/small.pt"
)


DATASET_DIR = Path(
    "/root/autodl-tmp/project/tongue_seg/dataset"
)


OUTPUT_DIR = Path(
    "/root/autodl-tmp/project/tongue_seg/output_roi_mask"
)


IMGSZ = 1024

CONF = 0.5


# ==========================
# 创建目录
# ==========================

for split in ["train", "val", "test"]:

    (
        OUTPUT_DIR /
        split /
        "images"
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================
# 加载模型
# ==========================

model = YOLO(MODEL_PATH)



# ==========================
# mask生成纯舌体ROI
# ==========================

def generate_mask_roi(img, mask):

    """
    img:
        原图

    mask:
        二值mask
        0背景
        1舌体
    """


    # 找mask外接框
    ys, xs = np.where(mask > 0)


    if len(xs) == 0:
        return None


    x1 = xs.min()
    y1 = ys.min()

    x2 = xs.max()
    y2 = ys.max()


    # 裁剪区域
    roi = img[
        y1:y2+1,
        x1:x2+1
    ]


    roi_mask = mask[
        y1:y2+1,
        x1:x2+1
    ]


    # mask三通道
    roi_mask = np.stack(
        [
            roi_mask,
            roi_mask,
            roi_mask
        ],
        axis=-1
    )


    # 舌体之外透明/黑色
    roi = roi * roi_mask


    return roi



# ==========================
# 推理生成
# ==========================

for split in ["train", "val", "test"]:


    img_dir = (
        DATASET_DIR /
        split /
        "images"
    )


    save_dir = (
        OUTPUT_DIR /
        split /
        "images"
    )


    for img_path in img_dir.glob("*"):


        img = cv2.imread(
            str(img_path)
        )


        if img is None:
            continue


        results = model.predict(
            source=img,
            imgsz=IMGSZ,
            conf=CONF,
            verbose=False
        )


        result = results[0]


        if result.masks is None:
            continue


        # 多mask情况取最大面积
        masks = result.masks.data.cpu().numpy()


        areas = [
            np.sum(m)
            for m in masks
        ]


        mask = masks[
            np.argmax(areas)
        ]


        # resize mask到原图尺寸
        mask = cv2.resize(
            mask,
            (
                img.shape[1],
                img.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )


        mask = (
            mask > 0.5
        ).astype(
            np.uint8
        )


        roi = generate_mask_roi(
            img,
            mask
        )


        if roi is None:
            continue


        save_path = (
            save_dir /
            img_path.name
        )


        cv2.imwrite(
            str(save_path),
            roi
        )


print("ROI生成完成")

# ==========================
# 推理生成
# ==========================

for split in ["train", "val", "test"]:


    img_dir = (
        DATASET_DIR /
        split /
        "images"
    )


    save_dir = (
        OUTPUT_DIR /
        split /
        "images"
    )


    img_list = list(
        img_dir.glob("*")
    )


    for img_path in tqdm(
            img_list,
            desc=f"Generating {split}",
            ncols=100
    ):


        img = cv2.imread(
            str(img_path)
        )


        if img is None:
            continue


        results = model.predict(
            source=img,
            imgsz=IMGSZ,
            conf=CONF,
            verbose=False
        )


        result = results[0]


        if result.masks is None:
            continue


        masks = (
            result.masks.data
            .cpu()
            .numpy()
        )


        # 最大mask作为舌体
        areas = [
            np.sum(m)
            for m in masks
        ]


        mask = masks[
            np.argmax(areas)
        ]


        mask = cv2.resize(
            mask,
            (
                img.shape[1],
                img.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )


        mask = (
            mask > 0.5
        ).astype(
            np.uint8
        )


        roi = generate_mask_roi(
            img,
            mask
        )


        if roi is None:
            continue


        save_path = (
            save_dir /
            img_path.name
        )


        cv2.imwrite(
            str(save_path),
            roi
        )


print("ROI生成完成")