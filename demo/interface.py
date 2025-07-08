import streamlit as st
import torch
import numpy as np
import cv2
from torchvision import transforms
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from utils import load_model, preprocess_image, predict_image, class_labels

st.set_page_config(page_title="EMNIST OCR Demo", layout="centered")

st.title("✍️ Nhận diện chữ viết tay - EMNIST")
st.markdown("Upload ảnh hoặc vẽ ký tự để mô hình dự đoán.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("model/emnist_cnn.pth", device)

# --- Upload ảnh ---
uploaded_file = st.file_uploader("📤 Tải ảnh lên (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Ảnh bạn đã upload", use_container_width=True)

    tensor = preprocess_image(image)
    label, conf = predict_image(model, tensor, device)
    st.success(f"🔍 Mô hình dự đoán: **{label}** (độ tin cậy: {conf:.2%})")

st.markdown("---")
st.subheader("🖌️ Hoặc thử **vẽ trực tiếp** bên dưới:")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    drawing_mode="freedraw",
    update_streamlit=True,
    height=280,
    width=280,
    key="canvas",
)

if st.button("📊 Dự đoán từ ảnh đã vẽ"):

    if canvas_result.image_data is not None:
        drawn_image = canvas_result.image_data  # (H, W, 4)
        drawn_pil = Image.fromarray((drawn_image).astype("uint8"), mode="RGBA").convert("RGB")

        st.subheader("Ảnh bạn đã vẽ:")
        st.image(drawn_pil, use_container_width=True)

    # 👉 Không cần threshold, đảo màu: chỉ resize + ToTensor
    transform_pipeline = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),  # model nhận ảnh 1 kênh
        transforms.ToTensor(),
    ])
    img_tensor = transform_pipeline(drawn_pil)
    img_tensor = torch.rot90(torch.fliplr(img_tensor), k=0, dims=[1, 2])
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, pred_class_index = torch.max(probabilities, 1)
        pred_index = pred_class_index.item()
        label = class_labels[pred_index] if pred_index < len(class_labels) else "Unknown"
        conf = confidence.item()

    st.success(f"🧠 Mô hình dự đoán: **{label}** (độ tin cậy: {conf:.2%})")
