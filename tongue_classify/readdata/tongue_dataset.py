"""
Tongue Dataset
多任务舌象分类

tasks:
    tongue_color
    coat_color
    coat_thickness
    crack
    teeth_mark
"""

from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TongueDataset(Dataset):

    REQUIRED_COLUMNS = [
        "image",
        "tongue_color",
        "coat_color",
        "coat_thickness",
        "crack",
        "teeth_mark"
    ]

    def __init__(
        self,
        excel_file,
        image_dir,
        img_size=300,
        mode="train"
    ):
        super().__init__()

        self.mode = mode
        self.img_size = img_size
        self.image_dir = Path(image_dir)

        # 读取 excel
        self.data = pd.read_excel(excel_file)

        # 检查字段
        self._check_columns()

        # ==========================
        # 数据增强
        # ==========================
        if self.mode == "train":
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),

                # 水平翻转
                transforms.RandomHorizontalFlip(p=0.5),

                # 垂直翻转（舌头图像偶尔有用）
                transforms.RandomVerticalFlip(p=0.2),

                # 旋转
                transforms.RandomRotation(degrees=15),

                # 平移 + 缩放
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1)
                ),

                # 颜色扰动（比原来更强）
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.05
                ),

                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            # 验证 / 测试 不做增强
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def _check_columns(self):
        missing = []
        for col in self.REQUIRED_COLUMNS:
            if col not in self.data.columns:
                missing.append(col)

        if missing:
            raise ValueError(f"Excel缺少字段: {missing}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # 图片路径
        img_name = row["image"]
        img_path = self.image_dir / img_name

        if not img_path.exists():
            raise FileNotFoundError(f"图片不存在: {img_path}")

        img = Image.open(img_path).convert("RGB")

        # 数据增强 / 预处理
        if self.transform:
            img = self.transform(img)

        # 多任务标签
        labels = {
            "tongue_color": torch.tensor(int(row["tongue_color"])).long(),
            "coat_color": torch.tensor(int(row["coat_color"])).long(),
            "coat_thickness": torch.tensor(int(row["coat_thickness"])).long(),
            "crack": torch.tensor(int(row["crack"])).long(),
            "teeth_mark": torch.tensor(int(row["teeth_mark"])).long()
        }

        return img, labels