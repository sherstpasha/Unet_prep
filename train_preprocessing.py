import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision

from dataset import (
    PreprocessingDataset,
)  # Датасет, возвращающий (input, target_pre) с resize_by_max_side
from model import UNet


# Функция для установки seed для воспроизводимости
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Используем модель UNet с features=16 (можете изменять в зависимости от экспериментов)
    model = UNet(in_channels=1, out_channels=1, features=32).to(device)

    lr = 1e-5
    epochs = 100
    batch_size = 2
    max_side = 1024  # Размер, по большей стороне которого масштабируются изображения

    # Пути к данным:
    input_dir = r"C:\data\images"  # входные изображения (grayscale)
    target_pre_dir = r"C:\data\gt"  # GT предобработки (бинарные маски)

    # Создаем датасет, который масштабирует изображения так, чтобы их большая сторона равнялась max_side
    # и дополняет до квадратного размера max_side x max_side.
    # Для тренировки можно включить аугментации (augment=True), для валидации – отключить (augment=False).
    full_dataset_train = PreprocessingDataset(
        input_dir, target_pre_dir, max_side=max_side, augment=True
    )
    full_dataset_val = PreprocessingDataset(
        input_dir, target_pre_dir, max_side=max_side, augment=False
    )

    total_samples = len(full_dataset_train)
    indices = list(range(total_samples))
    random.shuffle(indices)
    train_size = int(0.8 * total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_dataset_train, train_indices)
    val_dataset = Subset(full_dataset_val, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )
    bce_loss = nn.BCELoss()

    writer = SummaryWriter(log_dir="runs/preprocessing_experiment")
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss_epoch = 0.0
        for inputs, target_pre in train_loader:
            inputs = inputs.to(device)
            target_pre = target_pre.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = bce_loss(output, target_pre)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_epoch += loss.item() * inputs.size(0)
            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            global_step += 1
        train_loss_epoch /= len(train_dataset)

        # Логирование примера из обучающей выборки
        model.eval()
        with torch.no_grad():
            try:
                train_batch = next(iter(train_loader))
                train_inputs, train_targets = train_batch[0].to(device), train_batch[
                    1
                ].to(device)
                train_output = model(train_inputs)
                grid_train_inputs = torchvision.utils.make_grid(
                    train_inputs, nrow=2, normalize=True, scale_each=True
                )
                grid_train_output = torchvision.utils.make_grid(
                    train_output, nrow=2, normalize=True, scale_each=True
                )
                grid_train_target = torchvision.utils.make_grid(
                    train_targets, nrow=2, normalize=True, scale_each=True
                )
                writer.add_image("Train/Inputs", grid_train_inputs, epoch)
                writer.add_image("Train/Predicted", grid_train_output, epoch)
                writer.add_image("Train/Target", grid_train_target, epoch)
            except StopIteration:
                pass

        # Валидация
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for inputs, target_pre in val_loader:
                inputs = inputs.to(device)
                target_pre = target_pre.to(device)
                output = model(inputs)
                loss = bce_loss(output, target_pre)
                val_loss_epoch += loss.item() * inputs.size(0)
        val_loss_epoch /= len(val_dataset)

        writer.add_scalar("Loss/epoch_train", train_loss_epoch, epoch)
        writer.add_scalar("Loss/epoch_val", val_loss_epoch, epoch)

        # Логирование примера из валидационной выборки
        with torch.no_grad():
            try:
                val_batch = next(iter(val_loader))
                val_inputs, val_targets = val_batch[0].to(device), val_batch[1].to(
                    device
                )
                val_output = model(val_inputs)
                grid_val_inputs = torchvision.utils.make_grid(
                    val_inputs, nrow=2, normalize=True, scale_each=True
                )
                grid_val_output = torchvision.utils.make_grid(
                    val_output, nrow=2, normalize=True, scale_each=True
                )
                grid_val_target = torchvision.utils.make_grid(
                    val_targets, nrow=2, normalize=True, scale_each=True
                )
                writer.add_image("Val/Inputs", grid_val_inputs, epoch)
                writer.add_image("Val/Predicted", grid_val_output, epoch)
                writer.add_image("Val/Target", grid_val_target, epoch)
            except StopIteration:
                pass

        print(
            f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f}"
        )

        # Сохраняем лучшую модель по валидационному лоссу
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(model.state_dict(), "best_unet_preprocessing32.pth")
            print(
                f"Best model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}"
            )

    writer.close()
    print("✅ Preprocessing training completed!")
