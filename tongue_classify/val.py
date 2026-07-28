"""
export.py
多任务舌象分类模型全面评估脚本
生成混淆矩阵、各类指标、错误分析、总结报告
"""

import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from readdata.tongue_dataset import TongueDataset
from model.multitask import TongueMultiTaskModel
from torch.utils.data import DataLoader


# ==================================================
# 配置与工具
# ==================================================
TASKS = ["tongue_color", "coat_color", "coat_thickness", "crack", "teeth_mark"]

# 每个任务的中文名（方便看报告）
TASK_CN = {
    "tongue_color": "舌色",
    "coat_color": "苔色",
    "coat_thickness": "苔厚",
    "crack": "裂纹",
    "teeth_mark": "齿痕"
}


def load_config(path="configs/train.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==================================================
# 评估核心
# ==================================================
@torch.no_grad()
def collect_predictions(model, loader, device):
    """收集所有预测和真实标签"""
    model.eval()
    all_preds = {task: [] for task in TASKS}
    all_labels = {task: [] for task in TASKS}
    all_img_names = []

    for images, targets in tqdm(loader, desc="Inference"):
        images = images.to(device, non_blocking=True)
        outputs = model(images)

        for task in TASKS:
            preds = outputs[task].argmax(dim=1).cpu().numpy()
            labels = targets[task].cpu().numpy()
            all_preds[task].extend(preds)
            all_labels[task].extend(labels)

        # 如果 Dataset 返回了图片名，可以在这里收集（可选）
        # 当前 Dataset 没有返回文件名，后面用索引代替

    return all_preds, all_labels


def compute_metrics(y_true, y_pred, task_name):
    """计算单个任务的所有指标"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    return {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }


def plot_confusion_matrix(cm, task, save_path, class_names=None):
    """绘制并保存混淆矩阵热力图"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else range(cm.shape[1]),
        yticklabels=class_names if class_names else range(cm.shape[0]),
        cbar=True
    )
    plt.title(f"Confusion Matrix - {TASK_CN.get(task, task)} ({task})", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_metrics_bar(metrics_dict, save_path):
    """各任务 Accuracy 和 Macro-F1 对比柱状图"""
    tasks = list(metrics_dict.keys())
    accs = [metrics_dict[t]["accuracy"] for t in tasks]
    f1s = [metrics_dict[t]["f1_macro"] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width/2, f1s, width, label="Macro-F1", color="#DD8452")

    ax.set_ylabel("Score")
    ax.set_title("Per-Task Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_CN.get(t, t) for t in tasks], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # 在柱子上写数值
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def find_errors(y_true, y_pred, task, max_show=20):
    """找出预测错误的样本索引"""
    errors = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p:
            errors.append({
                "index": i,
                "true": int(t),
                "pred": int(p)
            })
            if len(errors) >= max_show:
                break
    return errors


# ==================================================
# 主函数
# ==================================================
def main():
    seed_everything(42)
    cfg = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 结果保存目录
    result_dir = Path("results/evaluation")
    result_dir.mkdir(parents=True, exist_ok=True)
    cm_dir = result_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)

    # ==========================
    # 加载测试集
    # ==========================
    test_dataset = TongueDataset(
        excel_file=cfg["dataset"]["test"],
        image_dir=cfg["dataset"]["test_image_dir"],
        img_size=cfg["dataset"]["img_size"],
        mode="val"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True
    )
    print(f"Test samples: {len(test_dataset)}")

    # ==========================
    # 加载模型
    # ==========================
    model = TongueMultiTaskModel(cfg)
    best_path = Path(cfg["checkpoint"]["save_dir"]) / "best.pt"

    if not best_path.exists():
        raise FileNotFoundError(f"找不到模型权重: {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model: {best_path}")

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")

    # ==========================
    # 推理
    # ==========================
    all_preds, all_labels = collect_predictions(model, test_loader, device)

    # ==========================
    # 计算指标 + 画图
    # ==========================
    all_metrics = {}
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("舌象多任务分类模型 - 测试集全面评估报告")
    summary_lines.append("=" * 60)
    summary_lines.append(f"模型权重      : {best_path}")
    summary_lines.append(f"测试样本数    : {len(test_dataset)}")
    summary_lines.append(f"模型参数量    : {total_params / 1e6:.2f}M")
    summary_lines.append(f"输入尺寸      : {cfg['dataset']['img_size']}")
    summary_lines.append("")

    for task in TASKS:
        y_true = np.array(all_labels[task])
        y_pred = np.array(all_preds[task])

        metrics = compute_metrics(y_true, y_pred, task)
        all_metrics[task] = metrics

        # 画混淆矩阵
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(
            cm,
            task,
            save_path=cm_dir / f"cm_{task}.png"
        )

        # 打印与记录
        cn_name = TASK_CN.get(task, task)
        summary_lines.append(f"【{cn_name}】({task})")
        summary_lines.append(f"  Accuracy      : {metrics['accuracy']:.4f}")
        summary_lines.append(f"  Macro-F1      : {metrics['f1_macro']:.4f}")
        summary_lines.append(f"  Weighted-F1   : {metrics['f1_weighted']:.4f}")
        summary_lines.append(f"  Macro-Precision: {metrics['precision_macro']:.4f}")
        summary_lines.append(f"  Macro-Recall  : {metrics['recall_macro']:.4f}")
        summary_lines.append(f"  Support       : {metrics['support']}")
        summary_lines.append("")
        summary_lines.append(metrics["classification_report"])
        summary_lines.append("-" * 50)

        # 错误样本
        errors = find_errors(y_true, y_pred, task, max_show=15)
        error_path = result_dir / f"errors_{task}.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)

    # 画综合对比图
    plot_metrics_bar(all_metrics, result_dir / "metrics_comparison.png")

    # 计算平均指标
    mean_acc = np.mean([all_metrics[t]["accuracy"] for t in TASKS])
    mean_f1 = np.mean([all_metrics[t]["f1_macro"] for t in TASKS])

    summary_lines.append("")
    summary_lines.append("=" * 60)
    summary_lines.append("【综合表现】")
    summary_lines.append(f"  平均 Accuracy : {mean_acc:.4f}")
    summary_lines.append(f"  平均 Macro-F1 : {mean_f1:.4f}")
    summary_lines.append("=" * 60)
    summary_lines.append("")
    summary_lines.append("结果文件说明：")
    summary_lines.append(f"  - 混淆矩阵图片 : {cm_dir}")
    summary_lines.append(f"  - 指标对比图   : {result_dir / 'metrics_comparison.png'}")
    summary_lines.append(f"  - 详细指标JSON : {result_dir / 'metrics.json'}")
    summary_lines.append(f"  - 错误样本     : {result_dir / 'errors_*.json'}")
    summary_lines.append(f"  - 本报告       : {result_dir / 'summary.txt'}")

    # 保存文件
    with open(result_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    with open(result_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # 终端打印
    print("\n".join(summary_lines))
    print(f"\n所有结果已保存到: {result_dir.resolve()}")


if __name__ == "__main__":
    main()