from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained("MiaoshouAI/Florence-2-large-PromptGen-v2.0", trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("MiaoshouAI/Florence-2-large-PromptGen-v2.0", trust_remote_code=True)

# Define the prompt
prompt = "<MORE_DETAILED_CAPTION>"

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # Get the image URL from the request
    data = request.json
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    try:
        # Download and open the image
        image = Image.open(requests.get(image_url, stream=True).raw)

        # Process the image and generate the caption
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

        # Return the generated caption
        return jsonify({"caption": parsed_answer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
