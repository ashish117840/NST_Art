# Neural Style Transfer Web App

A modern Streamlit web application for neural style transfer that transforms a content image using the style of another image. The app is hosted at:

https://neural-style-transfer-art.streamlit.app/

## Overview

This project implements an Adaptive Instance Normalization (AdaIN)-based neural style transfer pipeline using PyTorch. It allows users to upload a content image and a style image, then generate a stylized result directly in the browser.

### Key Features

- Upload a content image and a style image
- Choose from example images if needed
- Adjust the style strength with a slider
- Preview the generated stylized image
- Download the final result
- Modern dark-themed Streamlit UI

## Project Structure

```text
NST_CODE/
├── app.py                   # Streamlit application entry point
├── requirements.txt        # Python dependencies
├── train.py                 # Training script for the model
├── vgg_normalised.pth       # Pretrained VGG weights
├── experiment/
│   └── final_exp/
│       └── decoder_final.pth
├── utils/
│   ├── models.py            # Encoder and decoder model definitions
│   └── utils.py             # AdaIN and image utility functions
├── templates/              # Flask template files (legacy)
├── Demo_IO_Images/         # Example images for demo usage
├── static/uploads/         # Generated output images
└── tests/                  # Basic image processing tests
```

## Technologies Used

- Python
- PyTorch
- TorchVision
- Streamlit
- Pillow
- NumPy

## How It Works

The app uses a pretrained encoder-decoder architecture combined with AdaIN:

1. The content image and style image are resized and converted to tensors.
2. Feature maps are extracted from both images using the encoder.
3. Adaptive Instance Normalization is applied to match the style statistics.
4. The stylized features are combined with the content features.
5. The decoder reconstructs the final stylized image.

## Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/ashish117840/NST_Code.git
cd NST_Code
```

### 2. Create and activate a virtual environment

On Windows:

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

## Deployment

This project is deployed on Streamlit Cloud.

### Streamlit Cloud Deployment

To deploy your own copy:

1. Push this repository to GitHub.
2. Open Streamlit Cloud.
3. Create a new app from the GitHub repository.
4. Set the app file to `app.py`.
5. Deploy.

## Usage

- Upload a content image.
- Upload a style image.
- Adjust the style strength slider.
- Click the button to generate the stylized image.
- Download the resulting image.

## Notes

- The app uses the model weights stored in the repository.
- GPU support is used automatically when available.
- The generated images are saved in the `static/uploads` directory.

## License

This project is intended for educational and demonstration purposes.
