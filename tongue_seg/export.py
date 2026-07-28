# export.py


from ultralytics import YOLO



MODEL_PATH="weights/best.pt"



def main():


    model=YOLO(
        MODEL_PATH
    )


    model.export(

        format="onnx",

        imgsz=1024,

        opset=17,

        simplify=True

    )



if __name__=="__main__":

    main()