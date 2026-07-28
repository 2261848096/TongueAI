import random
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from readdata.tongue_dataset import TongueDataset
from model.multitask import TongueMultiTaskModel
from utils.loss import MultiTaskLoss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, amp):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc="Train")

    for images, targets in loop:
        images = images.to(device, non_blocking=True)
        for k in targets:
            targets[k] = targets[k].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp:
            with autocast():
                outputs = model(images)
                loss, loss_dict = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss, loss_dict = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Eval"):
    """同时返回 loss 和每个任务的准确率"""
    model.eval()
    total_loss = 0.0

    # 用于统计每个任务的正确数和总数
    correct = {task: 0 for task in ["tongue_color", "coat_color", "coat_thickness", "crack", "teeth_mark"]}
    total = {task: 0 for task in correct.keys()}

    for images, targets in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        for k in targets:
            targets[k] = targets[k].to(device, non_blocking=True)

        outputs = model(images)
        loss, _ = criterion(outputs, targets)
        total_loss += loss.item()

        # 计算每个任务的准确率
        for task in correct.keys():
            if task in outputs and task in targets:
                pred = outputs[task].argmax(dim=1)
                correct[task] += (pred == targets[task]).sum().item()
                total[task] += targets[task].size(0)

    avg_loss = total_loss / len(loader)
    acc = {task: correct[task] / total[task] if total[task] > 0 else 0.0 for task in correct}

    return avg_loss, acc


def main():
    cfg = load_config("configs/train.yaml")
    seed_everything(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ==========================
    # Dataset
    # ==========================
    train_dataset = TongueDataset(
        excel_file=cfg["dataset"]["train"],
        image_dir=cfg["dataset"]["train_image_dir"],
        img_size=cfg["dataset"]["img_size"],
        mode="train"
    )

    val_dataset = TongueDataset(
        excel_file=cfg["dataset"]["val"],
        image_dir=cfg["dataset"]["val_image_dir"],
        img_size=cfg["dataset"]["img_size"],
        mode="val"
    )

    test_dataset = TongueDataset(
        excel_file=cfg["dataset"]["test"],
        image_dir=cfg["dataset"]["test_image_dir"],
        img_size=cfg["dataset"]["img_size"],
        mode="val"          # test 也用 val 模式（不做增强）
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples  : {len(val_dataset)}")
    print(f"Test samples : {len(test_dataset)}")

    # ==========================
    # Model
    # ==========================
    model = TongueMultiTaskModel(cfg)
    model.to(device)

    # ==========================
    # Loss
    # ==========================
    criterion = MultiTaskLoss(cfg["loss"]["weights"])

    # ==========================
    # Optimizer
    # ==========================
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["optimizer"]["lr"],
        weight_decay=cfg["train"]["optimizer"]["weight_decay"]
    )

    # ==========================
    # Scheduler
    # ==========================
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["epochs"],
        eta_min=cfg["train"]["scheduler"]["min_lr"]
    )

    # AMP
    scaler = GradScaler(enabled=cfg["train"]["amp"])

    # ==========================
    # Checkpoint
    # ==========================
    save_dir = Path(cfg["checkpoint"]["save_dir"])
    save_dir.mkdir(exist_ok=True, parents=True)

    best_loss = float("inf")
    patience = cfg["early_stop"]["patience"]
    counter = 0

    # ==========================
    # Training loop
    # ==========================
    for epoch in range(cfg["train"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['train']['epochs']}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, cfg["train"]["amp"]
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc="Val")

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss  : {val_loss:.4f}")
        print("Val Acc   :", {k: f"{v:.4f}" for k, v in val_acc.items()})

        # 保存 best
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), save_dir / "best.pt")
            print("→ Save best model")
        else:
            counter += 1

        torch.save(model.state_dict(), save_dir / "last.pt")

        if counter >= patience:
            print("Early stopping triggered")
            break

    print(f"\nTraining finished. Best val loss: {best_loss:.4f}")

    # ==========================
    # Test Evaluation（用 best 模型）
    # ==========================
    print("\n" + "="*50)
    print("Start Test Evaluation with best model...")
    print("="*50)

    best_model_path = save_dir / "best.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Warning: best.pt not found, using current model")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc="Test")

    print(f"\nTest Loss: {test_loss:.4f}")
    print("Test Acc :")
    for task, acc in test_acc.items():
        print(f"  {task:15s}: {acc:.4f}")


if __name__ == "__main__":
    main()