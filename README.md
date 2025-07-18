# ✍️ EMNIST Handwriting OCR Project

A simple handwriting character recognition web app built with **PyTorch** and **Streamlit**, using the **EMNIST Balanced** dataset.

---

## 🚀 Features

✅ Recognize handwritten characters from:
- Uploaded images (jpg, png)
- Direct drawing on the web canvas

✅ Automatic binarization (white background, black text)

✅ CNN model predicts the character and shows confidence score

---

## 📊 Dataset
- Based on EMNIST Balanced, including 47 character classes.
- More info: https://www.nist.gov/itl/products-and-services/emnist-dataset

## ⚙️ Installation

### Clone this repository
```bash
git clone https://github.com/vuvanha2906/handwriting-ocr-project.git
cd handwriting-ocr-project
```
Install dependencies
```bash
pip install -r requirements.txt
```

## 🧠 Training the model

This project includes a Jupyter notebook to train the CNN model from scratch on CPU.

1️⃣ Open the notebook: 
```bash
cd notebooks
jupyter notebook train-model.ipynb
```

2️⃣ Run all cells to train your model (include download dataset).

3️⃣ After training, save the model as: `emnist_cnn.pth`, then copy `emnist_cnn.pth` into the `demo/model`.

4️⃣ You can test on notebooks

## ▶️ Run the app
```bash
cd demo
streamlit run interface.py
```

---

## ❤️ Contributing
- Feel free to open issues or pull requests to improve this project!


