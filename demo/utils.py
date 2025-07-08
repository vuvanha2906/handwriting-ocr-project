import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

from model.model import EMNIST_CNN

# Bảng class labels (47 lớp EMNIST balanced)
class_labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")


def load_model(path, device):
    model = EMNIST_CNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))

    resized_np = np.array(resized)
    threshold = 128
    binary_np = np.where(resized_np > threshold, 255, 0).astype(np.uint8)

    mean = np.mean(binary_np)
    if mean > 127:
        binary_np = 255 - binary_np

    resized = Image.fromarray(binary_np)
    # Đảo màu: chữ trắng, nền đen

    # Chuyển sang tensor
    tensor = transforms.ToTensor()(resized)

    # 🔁 Xoay ảnh: flip trái-phải + xoay 90 độ (nếu cần)
    tensor = torch.rot90(torch.fliplr(tensor), k=0, dims=[1, 2])

    return tensor.unsqueeze(0)  # Shape (1, 1, 28, 28)


def predict_image(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(prob, dim=1).item()
        confidence = prob[0, pred_idx].item()
        label = class_labels[pred_idx] if pred_idx < len(class_labels) else "?"
        return label, confidence
