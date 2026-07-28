import torch
import argparse

from PIL import Image
from torchvision import transforms

from model.multitask import TongueMultiTaskModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "/root/autodl-tmp/project/tongue_classify/checkpoints/best.pt"


TASK_NAMES = {
    "tongue_color": [
        "舌质淡白",
        "舌质淡红",
        "舌质红绛"
    ],
    "coat_color": [
        "舌苔白",
        "舌苔薄白",
        "舌苔黄"
    ],
    "coat_thickness": [
        "苔薄",
        "苔厚"
    ],
    "crack": [
        "无裂纹舌",
        "裂纹舌"
    ],
    "teeth_mark": [
        "无齿痕舌",
        "齿痕舌"
    ]
}


transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():
    # 按照 TongueMultiTaskModel 的要求构造 cfg
    cfg = {
        "model": {
            "pretrained": False,   # 预测时一般不需要再下载预训练权重
            "num_classes": {
                "tongue_color": 3,
                "coat_color": 3,
                "coat_thickness": 2,
                "crack": 2,
                "teeth_mark": 2
            }
        }
    }

    model = TongueMultiTaskModel(cfg)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(
        ckpt["model"] if "model" in ckpt else ckpt
    )

    model.to(DEVICE)
    model.eval()

    return model


@torch.no_grad()
def predict(img_path):
    model = load_model()

    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0).to(DEVICE)

    outputs = model(img)

    print("\n========== Prediction ==========\n")

    for task in TASK_NAMES:
        prob = torch.softmax(outputs[task], dim=1)
        score, index = torch.max(prob, dim=1)

        index = index.item()
        score = score.item()

        print(task)
        print("预测:", TASK_NAMES[task][index])
        print("概率:", f"{score:.4f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    predict(args.image)