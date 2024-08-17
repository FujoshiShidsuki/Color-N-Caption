
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load the model and tokenizer
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
    # Open the image
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    # Pre-process the image
    pixel_values = feature_extractor(images=i_image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate the caption
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    print("Caption:", preds)

def main():
    # Use tkinter to open a file dialog
    Tk().withdraw()  # Hide the root window
    image_path = askopenfilename(title="Select an image file", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        predict_image(image_path)

if __name__ == "__main__":
    main()
