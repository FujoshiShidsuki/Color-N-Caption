from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import sqlite3
import os

app = Flask(__name__)

# Load the model and tokenizer for image captioning
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set generation parameters
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_image(image_path):
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")
    pixel_values = feature_extractor(images=i_image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return preds[0]

def get_color_name(r, g, b):
    conn = sqlite3.connect('colors.db')
    cursor = conn.cursor()
    cursor.execute('''
    SELECT Color_Name FROM colors ORDER BY
    ((Red - ?) * (Red - ?) + (Green - ?) * (Green - ?) + (Blue - ?) * (Blue - ?)) ASC
    LIMIT 1
    ''', (r, r, g, g, b, b))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    else:
        return "Unknown Color"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle image file upload
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename != '':
                image_path = os.path.join('static', image_file.filename)
                image_file.save(image_path)
                caption = predict_image(image_path)
                img = Image.open(image_path)
                img_rgb = np.array(img.convert("RGB"))
                # Using the center pixel
                height, width, _ = img_rgb.shape
                center_pixel = img_rgb[height // 2, width // 2]
                r, g, b = center_pixel
                color_name = get_color_name(r, g, b)
                return render_template('index.html', caption=caption, color_name=color_name, image_path=image_file.filename)

    return render_template('index.html', caption=None, color_name=None, image_path=None)

@app.route('/color_name', methods=['GET'])
def color_name():
    r = int(request.args.get('r'))
    g = int(request.args.get('g'))
    b = int(request.args.get('b'))
    color_name = get_color_name(r, g, b)
    return jsonify(color_name)

if __name__ == '__main__':
    app.run(debug=True)
