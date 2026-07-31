# 🎨 Neural Style Transfer Web App

A modern Streamlit web application for neural style transfer — blend the content of one image with the artistic style of another, right in your browser.

**🔗 Live Demo:** [neural-style-transfer-art.streamlit.app](https://neural-style-transfer-art.streamlit.app/)
**📦 Repository:** [github.com/ashish117840/NST_Art](https://github.com/ashish117840/NST_Art)

---

## ✨ Overview

This project implements an **Adaptive Instance Normalization (AdaIN)**-based neural style transfer pipeline using PyTorch. Upload a content image and a style image, and the app generates a stylized result in seconds — no local setup required.

## 🚀 Key Features

- 🖼️ Upload a content image and a style image
- 🎯 Choose from built-in example images
- 🎚️ Adjust style strength with a live slider
- 👀 Preview the generated stylized image instantly
- ⬇️ Download the final result
- 🌙 Modern dark-themed Streamlit UI

## 🧠 How It Works

The app uses a pretrained encoder–decoder architecture combined with AdaIN:

1. **Preprocessing** — The content and style images are resized and converted to tensors.
2. **Feature Extraction** — An encoder extracts feature maps from both images.
3. **Style Transfer** — Adaptive Instance Normalization aligns the content features' statistics to match the style features.
4. **Blending** — Stylized features are combined with content features, weighted by the style-strength slider.
5. **Reconstruction** — A decoder reconstructs the final stylized image from the blended features.

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | PyTorch, TorchVision |
| Web Framework | Streamlit |
| Image Processing | Pillow, NumPy |

## 📁 Project Structure

```
NST_CODE/
├── app.py                     # Streamlit application entry point
├── requirements.txt           # Python dependencies
├── train.py                   # Training script for the model
├── vgg_normalised.pth         # Pretrained VGG weights
├── experiment/
│   └── final_exp/
│       └── decoder_final.pth  # Trained decoder weights
├── utils/
│   ├── models.py               # Encoder and decoder model definitions
│   └── utils.py                 # AdaIN and image utility functions
├── templates/                  # Flask template files (legacy)
├── Demo_IO_Images/             # Example images for demo usage
├── static/uploads/             # Generated output images
└── tests/                      # Basic image processing tests
```

## 💻 Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/ashish117840/NST_Art.git
cd NST_Art
```

### 2. Create and activate a virtual environment

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal — typically:

```
http://localhost:8501
```

## ☁️ Deployment

This project is deployed on [Streamlit Cloud](https://streamlit.io/cloud). To deploy your own copy:

1. Push this repository to GitHub.
2. Open [Streamlit Cloud](https://share.streamlit.io/).
3. Create a new app from your GitHub repository.
4. Set the app file to `app.py`.
5. Click **Deploy**.

## 📖 Usage

1. Upload a content image.
2. Upload a style image.
3. Adjust the style strength slider to taste.
4. Click the button to generate the stylized image.
5. Download the resulting image.

## 📝 Notes

- The app uses the model weights stored in the repository.
- GPU support is used automatically when available.
- Generated images are saved in the `static/uploads` directory.

## 📄 License

This project is intended for educational and demonstration purposes.
