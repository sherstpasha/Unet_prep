import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

from model import MultiStageModel, detection_loss
from dataset import MultiStageDataset


def draw_boxes_on_image(image, boxes, threshold=0.5, img_size=1024):
    """
    Рисует боксы на одном изображении.
      image: тензор [1, H, W] (значения 0-1)
      boxes: тензор [num_boxes, 5] (нормализованные: [objectness, x_center, y_center, width, height])
    Возвращает изображение с 3 каналами (uint8) с нарисованными боксовыми предсказаниями.
    """
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    image_uint8 = (image * 255).clamp(0, 255).to(torch.uint8)
    valid_indices = (boxes[:, 0] >= threshold).nonzero(as_tuple=False).squeeze(1)
    if valid_indices.numel() == 0:
        return image_uint8
    valid_boxes = boxes[valid_indices]
    x_center = valid_boxes[:, 1]
    y_center = valid_boxes[:, 2]
    box_w = valid_boxes[:, 3]
    box_h = valid_boxes[:, 4]
    x_min = (x_center - box_w / 2) * img_size
    y_min = (y_center - box_h / 2) * img_size
    x_max = (x_center + box_w / 2) * img_size
    y_max = (y_center + box_h / 2) * img_size
    boxes_pixel = torch.stack([x_min, y_min, x_max, y_max], dim=1).round().to(torch.int)
    drawn = torchvision.utils.draw_bounding_boxes(
        image_uint8, boxes_pixel, colors="green", width=2
    )
    return drawn


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_boxes = 20
    model = MultiStageModel(
        in_channels=1, out_channels=1, features=32, num_boxes=num_boxes
    ).to(device)

    # Флаг заморозки весов UNet
    freeze_unet = True
    if freeze_unet:
        unet_weights = torch.load("unet_preprocessing.pth", map_location=device)
        model.unet.load_state_dict(unet_weights)
        for param in model.unet.parameters():
            param.requires_grad = False
        print("UNet weights frozen.")

    lr = 1e-3
    epochs = 10
    batch_size = 2

    # Пути:
    input_dir = r"C:\data900\group_1"  # входные изображения
    target_pre_dir = r"C:\data900\group_1_processed"  # таргеты для предобработки
    target_box_dir = r"C:\data900\yolo_boxes"  # YOLO-аннотации (.txt) для боксов

    dataset = MultiStageDataset(
        input_dir,
        target_pre_dir,
        target_box_dir,
        img_size=(1024, 1024),
        max_boxes=num_boxes,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(dataloader), epochs=epochs
    )
    bce_loss = nn.BCELoss()

    writer = SummaryWriter(log_dir="runs/multistage_detector_experiment")
    global_step = 0

    for epoch in range(epochs):
        model.train()
        for inputs, target_pre, boxes_target in dataloader:
            inputs = inputs.to(device)
            target_pre = target_pre.to(device)
            boxes_target = boxes_target.to(device)

            optimizer.zero_grad()
            # Модель возвращает два выхода:
            # 1) предобработанное изображение (UNet output)
            # 2) предсказанные боксы (детекционная голова)
            preprocessed, pred_boxes = model(inputs)
            loss_pre = bce_loss(preprocessed, target_pre)
            loss_det = detection_loss(pred_boxes, boxes_target)
            loss = loss_pre + loss_det
            loss.backward()
            optimizer.step()
            scheduler.step()

            writer.add_scalar("Loss/train_pre", loss_pre.item(), global_step)
            writer.add_scalar("Loss/train_det", loss_det.item(), global_step)
            writer.add_scalar("Loss/train_total", loss.item(), global_step)
            global_step += 1

        # Логирование изображений последнего батча текущей эпохи:
        grid_inputs = torchvision.utils.make_grid(
            inputs, nrow=2, normalize=True, scale_each=True
        )
        grid_preprocessed = torchvision.utils.make_grid(
            preprocessed, nrow=2, normalize=True, scale_each=True
        )
        grid_target_pre = torchvision.utils.make_grid(
            target_pre, nrow=2, normalize=True, scale_each=True
        )
        writer.add_image("Inputs", grid_inputs, epoch)
        writer.add_image("Predicted Preprocessed", grid_preprocessed, epoch)
        writer.add_image("Target Preprocessed", grid_target_pre, epoch)

        # Для детекционной головы: отрисовка боксов предсказаний и таргетов на соответствующих изображениях.
        drawn_pred = []
        drawn_target = []
        for i in range(pred_boxes.shape[0]):
            img_pred = preprocessed[i].detach().cpu()  # [1, H, W]
            pred_b = pred_boxes[i].detach().cpu()  # [num_boxes, 5]
            target_b = boxes_target[i].detach().cpu()  # [num_boxes, 5]
            drawn_pred.append(
                draw_boxes_on_image(img_pred, pred_b, threshold=0.5, img_size=1024)
            )
            drawn_target.append(
                draw_boxes_on_image(
                    target_pre[i].detach().cpu(), target_b, threshold=0.5, img_size=1024
                )
            )
        if drawn_pred:
            grid_pred_boxes = torchvision.utils.make_grid(
                torch.stack(drawn_pred), nrow=2
            )
            writer.add_image("Predicted Detection Boxes", grid_pred_boxes, epoch)
        if drawn_target:
            grid_target_boxes = torchvision.utils.make_grid(
                torch.stack(drawn_target), nrow=2
            )
            writer.add_image("Target Detection Boxes", grid_target_boxes, epoch)

        writer.add_scalar("Loss/epoch", loss.item(), epoch)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "multistage_detector.pth")
    writer.close()
    print("✅ Multistage training completed!")
