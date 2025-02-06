import os
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from flask import Flask, request, jsonify
from concurrent.futures import ProcessPoolExecutor

# Initialize Flask app
app = Flask(__name__)

# Load the model and processor
model = AutoModelForCausalLM.from_pretrained("MiaoshouAI/Florence-2-large-PromptGen-v2.0", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("MiaoshouAI/Florence-2-large-PromptGen-v2.0", trust_remote_code=True)

# Prepare the device (GPU if available, else CPU)
device = torch.device("cuda")
model.to(device)

# Helper function to decode an image from base64
def decode_image_from_base64(base64_string):
    # Decode the image from the base64 string
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

# Helper function to process images
def process_image(image, prompt):
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device)

    generated_ids = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

    return parsed_answer

# Define the endpoint for caption generation
@app.route("/caption", methods=["POST"])
def generate_caption():
    try:
        # Get JSON data from the request
        data = request.get_json()
        prompt = data.get("prompt", "<MORE_DETAILED_CAPTION>")  # Default prompt if none is provided
        image_base64 = data.get("image_base64")

        if not image_base64:
            return jsonify({"error": "Image base64 string is required"}), 400

        # Decode the image from base64
        image = decode_image_from_base64(image_base64)
        caption = process_image(image, prompt)

        # Return the result as a JSON response
        return jsonify({"caption": caption})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

