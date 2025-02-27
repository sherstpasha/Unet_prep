import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


class MultiStageDataset(Dataset):
    def __init__(
        self,
        input_dir,
        target_pre_dir,
        target_box_dir,
        img_size=(1024, 1024),
        max_boxes=20,
    ):
        """
        Параметры:
          input_dir: папка с входными изображениями.
          target_pre_dir: папка с таргетами для предобработки (ground truth для UNet).
          target_box_dir: папка с YOLO-аннотациями (.txt) для боксов.
          img_size: целевой размер (ширина, высота).
          max_boxes: максимальное число боксов (для паддинга).
        """
        self.input_files = sorted(os.listdir(input_dir))
        self.input_dir = input_dir
        self.target_pre_dir = target_pre_dir
        self.target_box_dir = target_box_dir
        self.img_size = img_size
        self.max_boxes = max_boxes

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        image_name = self.input_files[idx]

        # Загрузка входного изображения (грейскейл)
        input_path = os.path.join(self.input_dir, image_name)
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load image {input_path}")
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1, H, W)

        # Загрузка таргета для предобработки
        target_pre_path = os.path.join(self.target_pre_dir, image_name)
        pre_img = cv2.imread(target_pre_path, cv2.IMREAD_GRAYSCALE)
        if pre_img is None:
            raise ValueError(f"Cannot load preprocessed target {target_pre_path}")
        pre_img = cv2.resize(pre_img, self.img_size, interpolation=cv2.INTER_AREA)
        pre_img = pre_img.astype(np.float32) / 255.0
        pre_img = np.expand_dims(pre_img, axis=0)

        # Загрузка боксов из .txt файла
        base_name = os.path.splitext(image_name)[0]
        label_path = os.path.join(self.target_box_dir, base_name + ".txt")
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    # Если формат: либо 4 числа (x_center, y_center, width, height), либо 5 чисел (class, x_center, y_center, width, height)
                    if len(parts) == 4:
                        x_center, y_center, box_w, box_h = map(float, parts)
                    elif len(parts) == 5:
                        x_center, y_center, box_w, box_h = map(float, parts[1:])
                    else:
                        continue
                    boxes.append([1.0, x_center, y_center, box_w, box_h])
        boxes = np.array(boxes, dtype=np.float32)
        if boxes.size == 0:
            boxes = np.zeros((0, 5), dtype=np.float32)
        num_boxes = boxes.shape[0]
        if num_boxes < self.max_boxes:
            pad = np.zeros((self.max_boxes - num_boxes, 5), dtype=np.float32)
            boxes = np.concatenate([boxes, pad], axis=0)
        else:
            boxes = boxes[: self.max_boxes, :]

        img_tensor = torch.tensor(img, dtype=torch.float32)
        pre_tensor = torch.tensor(pre_img, dtype=torch.float32)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        return img_tensor, pre_tensor, boxes_tensor


def resize_by_max_side(img, max_side):
    """
    Масштабирует изображение так, чтобы его большая сторона стала равной max_side,
    а затем дополняет (padding) его до квадратного размера (max_side x max_side),
    сохраняя исходное соотношение сторон.
    """
    h, w = img.shape
    if w >= h:
        new_w = max_side
        new_h = int(h * max_side / w)
    else:
        new_h = max_side
        new_w = int(w * max_side / h)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_h = max_side - new_h
    pad_w = max_side - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )
    return padded


class PreprocessingDataset(Dataset):
    def __init__(self, input_dir, target_dir, max_side=1024, augment=True):
        """
        Параметры:
          input_dir (str): папка с входными изображениями (grayscale).
          target_dir (str): папка с GT масками (grayscale, бинарные).
          max_side (int): размер большей стороны после масштабирования (окончательный размер изображения будет квадратным max_side x max_side).
          augment (bool): если True, применяются аугментации.
        """
        self.input_files = sorted(os.listdir(input_dir))
        self.target_files = sorted(os.listdir(target_dir))
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.max_side = max_side

        # Аугментации с Albumentations с рекомендованными вероятностями:
        if augment:
            self.augment = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                    A.CoarseDropout(
                        max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.3
                    ),
                ],
                additional_targets={"mask": "mask"},
            )
        else:
            self.augment = None

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])
        input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if input_img is None or target_img is None:
            raise ValueError(f"Ошибка загрузки: {input_path} или {target_path}")

        # Масштабирование и паддинг по большей стороне
        input_img = resize_by_max_side(input_img, self.max_side)
        target_img = resize_by_max_side(target_img, self.max_side)

        # Применяем аугментации, если включены
        if self.augment is not None:
            augmented = self.augment(image=input_img, mask=target_img)
            input_img = augmented["image"]
            target_img = augmented["mask"]

        # Нормализация
        input_img = input_img.astype(np.float32) / 255.0
        target_img = target_img.astype(np.float32) / 255.0

        # Добавляем размерность канала: (H, W) -> (1, H, W)
        input_img = np.expand_dims(input_img, axis=0)
        target_img = np.expand_dims(target_img, axis=0)

        input_tensor = torch.tensor(input_img, dtype=torch.float32)
        target_tensor = torch.tensor(target_img, dtype=torch.float32)
        return input_tensor, target_tensor
