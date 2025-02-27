import os
import cv2
import torch
import numpy as np
from model import UNet


def load_model(model_path, device):
    # Создаем модель с теми же параметрами, что и при обучении
    model = UNet(in_channels=1, out_channels=1, features=16)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def inference(model, input_folder, output_folder, img_size=(1024, 1024), threshold=0.5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = sorted(os.listdir(input_folder))
    for image_name in image_files:
        image_path = os.path.join(input_folder, image_name)
        # Загружаем изображение в grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Не удалось загрузить {image_path}. Пропускаем.")
            continue

        # Изменяем размер изображения
        img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        # Приводим к float32 и нормализуем
        img_norm = img_resized.astype(np.float32) / 255.0
        # Добавляем размерность канала: (H, W) -> (1, H, W)
        img_tensor = (
            torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
        )  # [1, 1, H, W]
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            output = model(img_tensor)  # [1, 1, H, W]

        # Убираем размерность батча и канала: [H, W]
        output_mask = output.squeeze().cpu().numpy()
        # Применяем порог, чтобы получить бинарную маску
        binary_mask = (output_mask > threshold).astype(np.uint8) * 255

        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, binary_mask)
        print(f"Сохранено предсказание для {image_name} в {output_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = (
        "best_unet_preprocessing16.pth"  # Лучшие веса, сохраненные во время обучения
    )
    model = load_model(model_path, device)

    input_folder = r"C:\for_test"  # Папка с входными изображениями
    output_folder = r"C:\pred_masks"  # Папка для сохранения предсказанных масок

    inference(model, input_folder, output_folder, img_size=(1024, 1024), threshold=0.5)
    print("✅ Inference completed!")
