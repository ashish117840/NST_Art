import os
from pathlib import Path
from uuid import uuid4

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from utils.models import Decoder, VGGEncoder
from utils.utils import adaptive_instance_normalization


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
CONTENT_EXAMPLE_DIR = BASE_DIR / "Demo_IO_Images" / "i-p"
STYLE_EXAMPLE_DIR = BASE_DIR / "Demo_IO_Images" / "o-p"
MODEL_PATH = BASE_DIR / "experiment" / "final_exp" / "decoder_final.pth"
VGG_PATH = BASE_DIR / "vgg_normalised.pth"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@st.cache_resource(show_spinner=False)
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VGGEncoder(str(VGG_PATH)).to(device)
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    encoder.eval()
    decoder.eval()
    torch.set_grad_enabled(False)
    return encoder, decoder, device


def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    image_size = 256

    content_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    style_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    content_tensor = content_transform(content_image).unsqueeze(0).to(device)
    style_tensor = style_transform(style_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        content_feats = encoder(content_tensor, is_test=True)
        style_feats = encoder(style_tensor, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)
        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats
        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.detach().cpu().clone().squeeze(0)
    image = image.clamp(0, 1)
    pil_image = transforms.ToPILImage()(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(path)


def list_image_files(folder):
    if not folder.exists():
        return []
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )


def load_pil_image(path_or_file):
    if isinstance(path_or_file, (str, Path)):
        with Image.open(path_or_file) as img:
            return img.convert("RGB")

    if path_or_file is None:
        return None

    image = Image.open(path_or_file)
    return image.convert("RGB")


def main():
    st.set_page_config(page_title="Neural Style Transfer", page_icon="🎨", layout="wide")

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.98)),
                        url('https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?q=80&w=1974&auto=format&fit=crop');
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: #e2e8f0;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 4rem;
            max-width: 1400px;
        }
        .hero {
            text-align: center;
            padding: 3rem 0 2rem;
        }
        .hero h1 {
            font-family: 'Orbitron', 'Segoe UI', sans-serif;
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .hero p {
            color: #cbd5e1;
            font-size: 1.1rem;
            max-width: 700px;
            margin: 0 auto;
        }
        .card {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 1.2rem;
            box-shadow: 0 20px 45px -20px rgba(0,0,0,0.55);
            backdrop-filter: blur(12px);
            margin-bottom: 1rem;
        }
        .card h3, .card h4 {
            font-family: 'Orbitron', 'Segoe UI', sans-serif;
            color: #f8fafc;
            margin-bottom: 0.5rem;
        }
        .card .stImage img {
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.15);
            background: #0f172a;
        }
        div[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.85);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        .stButton > button {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            color: white;
            border: none;
            border-radius: 999px;
            padding: 0.7rem 1.2rem;
            font-weight: 600;
        }
        .stButton > button:hover {
            box-shadow: 0 10px 20px -10px rgba(168, 85, 247, 0.7);
        }
        .preview-box {
            min-height: 320px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #94a3b8;
            border: 2px dashed #334155;
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.7);
            padding: 1rem;
        }
        .result-box {
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 16px;
            padding: 1rem;
            background: rgba(5, 150, 105, 0.15);
        }
        .example-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1rem;
        }
        .example-item {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 0.8rem;
        }
        .example-item img {
            border-radius: 10px;
            width: 100%;
            height: 180px;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>STYLEFORGE AI</h1>
            <p>Redefine Reality with AI-Powered Artistry</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    encoder, decoder, device = load_models()

    content_examples = list_image_files(CONTENT_EXAMPLE_DIR)
    style_examples = list_image_files(STYLE_EXAMPLE_DIR)

    with st.sidebar:
        st.markdown("<h3 style='color:#f8fafc;'>Inputs</h3>", unsafe_allow_html=True)
        content_source = st.radio("Content image source", ["Upload", "Example"], horizontal=True)
        if content_source == "Upload":
            content_file = st.file_uploader("Choose content image", type=["png", "jpg", "jpeg"])
            selected_content = None
        else:
            content_file = None
            selected_content = st.selectbox(
                "Choose a content example",
                content_examples or [None],
                format_func=lambda path: path.name if path else "No example found",
            )

        style_source = st.radio("Style image source", ["Upload", "Example"], horizontal=True, key="style_source")
        if style_source == "Upload":
            style_file = st.file_uploader("Choose style image", type=["png", "jpg", "jpeg"], key="style_upload")
            selected_style = None
        else:
            style_file = None
            selected_style = st.selectbox(
                "Choose a style example",
                style_examples or [None],
                format_func=lambda path: path.name if path else "No example found",
                key="style_example",
            )

        alpha = st.slider("Blend strength", 0.0, 1.0, 1.0, 0.05)
        run_button = st.button("Generate stylized image", use_container_width=True)

    if content_source == "Upload":
        content_image = load_pil_image(content_file)
    else:
        content_image = load_pil_image(selected_content) if selected_content else None

    if style_source == "Upload":
        style_image = load_pil_image(style_file)
    else:
        style_image = load_pil_image(selected_style) if selected_style else None

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3>Content Source</h3></div>', unsafe_allow_html=True)
        if content_image is not None:
            st.image(content_image, width=700)
        else:
            st.markdown('<div class="preview-box">Select a content image</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><h3>Style Reference</h3></div>', unsafe_allow_html=True)
        if style_image is not None:
            st.image(style_image, width=700)
        else:
            st.markdown('<div class="preview-box">Select a style image</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3 style="text-align:center;">Style Strength</h3></div>', unsafe_allow_html=True)
    st.slider("", 0.0, 1.0, alpha, 0.05, key="alpha_display")

    if run_button:
        if content_image is None or style_image is None:
            st.error("Please provide both a content image and a style image.")
        else:
            with st.spinner("Applying style transfer..."):
                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

            output_path = UPLOAD_DIR / f"stylized_{uuid4().hex[:8]}.png"
            save_image(stylized_image, output_path)

            st.markdown('<div class="card"><h3>Stylized Result</h3></div>', unsafe_allow_html=True)
            st.image(output_path, width=700)
            with open(output_path, "rb") as output_file:
                st.download_button(
                    "Download result",
                    output_file,
                    file_name=output_path.name,
                    mime="image/png",
                )

    st.markdown('<div class="card"><h3>Examples</h3></div>', unsafe_allow_html=True)
    example_cols = st.columns(2)
    for idx, col in enumerate(example_cols):
        with col:
            st.markdown('<div class="example-item">', unsafe_allow_html=True)
            if idx == 0 and content_examples:
                st.image(content_examples[0], caption="Content example")
            elif idx == 1 and style_examples:
                st.image(style_examples[0], caption="Style example")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>FAQ</h3></div>', unsafe_allow_html=True)
    with st.expander("1. Is it a pretrained model?"):
        st.write("No, we train a model ourselves.")
    with st.expander("2. Is it a free platform?"):
        st.write("This demo is currently available as a free experience.")
    with st.expander("3. Which styles of painting can be used?"):
        st.write("You can use almost any painting style image as the style reference.")


if __name__ == "__main__":
    main()





