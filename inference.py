import os
import cv2
import numpy as np
import torch
from model import MultiStageModel


def draw_boxes(image, boxes, threshold=0.5):
    """
    Рисует боксы на изображении.
      image: цветное изображение (BGR) размером (H, W, 3)
      boxes: массив (num_boxes, 5) [objectness, x_center, y_center, width, height] (нормализованные)
    """
    H, W = image.shape[:2]
    for box in boxes:
        score = box[0]
        if score > threshold:
            x_center, y_center, box_w, box_h = box[1:]
            x_center_abs = int(x_center * W)
            y_center_abs = int(y_center * H)
            box_w_abs = int(box_w * W)
            box_h_abs = int(box_h * H)
            left = int(x_center_abs - box_w_abs / 2)
            top = int(y_center_abs - box_h_abs / 2)
            right = int(x_center_abs + box_w_abs / 2)
            bottom = int(y_center_abs + box_h_abs / 2)
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    return image


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_boxes = 20
    model = MultiStageModel(
        in_channels=1, out_channels=1, features=32, num_boxes=num_boxes
    ).to(device)
    model.load_state_dict(torch.load("multistage_detector.pth", map_location=device))
    model.eval()

    input_folder = r"C:\data900\inputs"
    output_folder = r"C:\data900\detector_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_resized = cv2.resize(img, (1024, 1024))
        img_norm = img_resized.astype(np.float32) / 255.0
        img_tensor = (
            torch.tensor(img_norm, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            preprocessed, pred_boxes = model(img_tensor)
        pred_boxes = pred_boxes[0].cpu().numpy()  # (num_boxes, 5)

        # Для рисования боксов: используем предобработанное изображение, переводим в цветное (BGR)
        pre_img = preprocessed[0].cpu().numpy()  # (H, W)
        pre_img = np.clip(pre_img * 255.0, 0, 255).astype(np.uint8)
        pre_img_color = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2BGR)
        img_with_boxes = draw_boxes(pre_img_color, pred_boxes, threshold=0.5)
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, img_with_boxes)
        print(f"Saved {out_path}")
    print("✅ Inference completed!")
