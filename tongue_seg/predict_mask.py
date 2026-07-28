from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path



# ==========================
# 配置
# ==========================


MODEL_PATH = (
    "/root/autodl-tmp/project/tongue_seg/models/weights/small.pt"
)


DATASET_DIR = Path(
    "/root/autodl-tmp/project/tongue_seg/dataset"
)


OUTPUT_DIR = Path(
    "/root/autodl-tmp/project/tongue_seg/output_mask"
)


IMGSZ = 1024


CONF = 0.5



# mask扩大参数
# 小范围扩大，保留边界信息

KERNEL_SIZE = 15



# ==========================
# 创建目录
# ==========================

for split in [
    "train",
    "val",
    "test"
]:

    (
        OUTPUT_DIR
        /
        split
        /
        "masks"
    ).mkdir(
        parents=True,
        exist_ok=True
    )





# ==========================
# mask生成
# ==========================

def process_image(
        model,
        img_path,
        save_path
):


    img=cv2.imread(
        str(img_path)
    )


    if img is None:

        return



    h,w,_=img.shape



    result=model.predict(

        source=str(img_path),

        imgsz=IMGSZ,

        conf=CONF,

        device=0,

        verbose=False

    )[0]



    # 没有mask

    if result.masks is None:


        print(
            "未检测:",
            img_path.name
        )

        return




    # ======================
    # 获取最大mask
    # ======================

    masks=result.masks.data


    max_area=0

    best_mask=None



    for m in masks:


        m=m.cpu().numpy()


        area=m.sum()



        if area > max_area:

            max_area=area

            best_mask=m




    if best_mask is None:

        return




    # ======================
    # resize mask
    # ======================

    mask=(

        best_mask*255

    ).astype(
        np.uint8
    )


    mask=cv2.resize(

        mask,

        (w,h),

        interpolation=cv2.INTER_NEAREST

    )



    # ======================
    # mask轻微扩大
    # ======================


    kernel=np.ones(

        (
            KERNEL_SIZE,
            KERNEL_SIZE
        ),

        np.uint8

    )


    mask=cv2.dilate(

        mask,

        kernel,

        iterations=1

    )



    # 二值化

    mask=np.where(

        mask>127,

        255,

        0

    ).astype(
        np.uint8
    )



    # 保存

    cv2.imwrite(

        str(save_path),

        mask

    )



    print(
        "完成:",
        img_path.name
    )







# ==========================
# 主程序
# ==========================


def main():


    print(
        "加载YOLO-seg..."
    )


    model=YOLO(

        MODEL_PATH

    )



    for split in [

        "train",

        "val",

        "test"

    ]:


        img_dir=(

            DATASET_DIR
            /
            split
            /
            "images"

        )


        save_dir=(

            OUTPUT_DIR
            /
            split
            /
            "masks"

        )



        images=list(

            img_dir.glob("*.*")

        )



        print(

            split,

            len(images)

        )



        for img in images:


            save_path=(

                save_dir
                /
                (
                    img.stem+".png"
                )

            )



            process_image(

                model,

                img,

                save_path

            )



    print(

        "全部mask生成完成"

    )




if __name__=="__main__":

    main()