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
    "/root/autodl-tmp/project/tongue_seg/output_roiContext"
)


IMGSZ = 1024

CONF = 0.5


# bbox扩大比例
PADDING_X = 0.07   # 横向扩大15%
PADDING_Y = 0.05   # 纵向扩大10%



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
# bbox扩大
# ==========================

def expand_bbox(
        x1,
        y1,
        x2,
        y2,
        w,
        h,
        ratio_x=0.15,
        ratio_y=0.10
):

    bw = x2 - x1
    bh = y2 - y1


    pad_x = int(
        bw * ratio_x
    )

    pad_y = int(
        bh * ratio_y
    )


    x1 = max(
        0,
        x1 - pad_x
    )


    y1 = max(
        0,
        y1 - pad_y
    )


    x2 = min(
        w,
        x2 + pad_x
    )


    y2 = min(
        h,
        y2 + pad_y
    )


    return x1,y1,x2,y2





# ==========================
# 单张生成ROI
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

        print(
            "读取失败:",
            img_path
        )

        return



    h,w,_ = img.shape



    # ======================
    # YOLO-seg预测
    # ======================

    result=model.predict(

        source=str(img_path),

        imgsz=IMGSZ,

        conf=CONF,

        device=0,

        verbose=False

    )[0]



    # 没检测到

    if result.masks is None:

        print(
            "未检测:",
            img_path.name
        )

        return




    # ======================
    # 选择最大mask
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
    # mask处理
    # ======================

    mask=(

        best_mask * 255

    ).astype(
        np.uint8
    )



    # resize到原图尺寸
    mask=cv2.resize(

        mask,

        (w,h),

        interpolation=cv2.INTER_NEAREST

    )



    # ======================
    # mask膨胀
    # 防止边缘细节丢失
    # ======================

    kernel=np.ones(

        (15,15),

        np.uint8

    )


    mask=cv2.dilate(

        mask,

        kernel,

        iterations=1

    )




    # ======================
    # 根据mask计算bbox
    # ======================

    ys,xs=np.where(

        mask > 0

    )


    if len(xs)==0:

        return



    x1=xs.min()

    y1=ys.min()

    x2=xs.max()

    y2=ys.max()




    # ======================
    # bbox扩大
    # ======================

    x1,y1,x2,y2 = expand_bbox(

        x1,

        y1,

        x2,

        y2,

        w,

        h,

        PADDING_X,

        PADDING_Y

    )




    # ======================
    # crop ROI
    # ======================

    roi=img[

        y1:y2,

        x1:x2

    ]



    if roi.size == 0:

        return




    # ======================
    # 保存RGBA PNG
    #
    # 保留:
    # 舌体
    # 嘴唇
    # 口腔环境
    #
    # 不透明背景
    # ======================


    rgba=np.zeros(

        (

            roi.shape[0],

            roi.shape[1],

            4

        ),

        dtype=np.uint8

    )



    # RGB

    rgba[:,:,:3]=roi



    # alpha全部保留

    rgba[:,:,3]=255




    cv2.imwrite(

        str(save_path),

        rgba

    )



    print(

        "完成:",

        img_path.name

    )







# ==========================
# 主函数
# ==========================

def main():


    print(
        "加载模型..."
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
            "images"

        )



        images=list(

            img_dir.glob("*.*")

        )



        print(

            split,

            "数量:",

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

        "全部ROI生成完成"

    )





if __name__=="__main__":

    main()