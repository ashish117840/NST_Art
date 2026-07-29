import os
from pathlib import Path
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
import io

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

EXAMPLE_DIRECTORIES = [
    Path('examples'),
    Path('Demo_IO_Images'),
    Path('Demo_IO_Images/i-p'),
    Path('Demo_IO_Images/o-p'),
    Path('content_data'),
    Path('style_data'),
    Path('static/uploads'),
    Path('experiment/final_exp'),
]

EXAMPLE_OUTPUT_SPECS = {
    'stylized_brad_pitt.jpg': ('brad_pitt.jpg', 'sketch.png'),
    'stylized_brad_pitt (1).jpg': ('brad_pitt.jpg', 'picasso_seated_nude_hr.jpg'),
}

EXAMPLE_OUTPUT_DIR = Path(app.config['UPLOAD_FOLDER']) / 'examples'


def resolve_example_path(filename):
    filename = Path(filename).name

    for base_dir in EXAMPLE_DIRECTORIES:
        candidate = base_dir / filename
        if candidate.exists() and candidate.is_file():
            return candidate

    for base_dir in EXAMPLE_DIRECTORIES:
        for path in base_dir.rglob(filename):
            if path.is_file():
                return path

    return None


def build_example_output(filename):
    filename = Path(filename).name

    if filename not in EXAMPLE_OUTPUT_SPECS:
        return None

    content_name, style_name = EXAMPLE_OUTPUT_SPECS[filename]
    content_path = resolve_example_path(content_name)
    style_path = resolve_example_path(style_name)

    if content_path is None or style_path is None:
        return None

    EXAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXAMPLE_OUTPUT_DIR / filename

    if not output_path.exists():
        content_image = Image.open(content_path).convert('RGB')
        style_image = Image.open(style_path).convert('RGB')
        stylized_image = style_transfer(content_image, style_image, encoder, decoder, 1.0, device)
        save_image(stylized_image, output_path)

    return output_path


class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VGGEncoder('vgg_normalised.pth').to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load('D:/Apna Collage/AI Proj/NST_CODE/experiment/final_exp/decoder_final.pth', map_location=device))

encoder.eval()
decoder.eval()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)



@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None
    submitted = request.method == 'POST'

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data

        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            
            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')

                alpha = float(form.alpha.data)
                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

                result_filename = 'stylized_' + content_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)
                
                result_image = result_filename
            except Exception as e:
                error = str(e)
    elif submitted:
        if not (form.content.data and form.content.data.filename) and not form.content_path.data:
            error = 'Please upload content image'
        elif not (form.style.data and form.style.data.filename) and not form.style_path.data:
            error = 'Please upload style image'

    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
                           style_image=style_filename, error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    generated_output = build_example_output(filename)
    if generated_output is not None:
        return send_from_directory(generated_output.parent, generated_output.name)

    file_path = resolve_example_path(filename)
    if file_path is None:
        abort(404)
    return send_from_directory(file_path.parent, file_path.name)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)





