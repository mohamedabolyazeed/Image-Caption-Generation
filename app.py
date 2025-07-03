from flask import Flask, render_template, request
from src.generate_caption import load_models, generate_caption
import os

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')

# Load models once when the app starts
caption_model, tokenizer, feature_extractor = load_models()

# Default image URL (empty initially)
default_image_url = 'uploaded_image.jpg'

@app.route('/')
def index():
    # Check if the image exists in the static folder
    image_path = os.path.join('web/static', default_image_url)
    image_url = default_image_url if os.path.exists(image_path) else None
    return render_template('index.html', image_url=image_url)

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return "No image uploaded", 400
    image = request.files['image']
    if image.filename == '':
        return "No image selected", 400
    
    # Save the image in the static folder
    image_path = os.path.join('web/static', 'uploaded_image.jpg')
    image.save(image_path)
    
    # Generate caption
    caption = generate_caption(image_path, caption_model, tokenizer, feature_extractor)
    
    # Pass the image path (relative to static folder) and caption to the template
    image_url = 'uploaded_image.jpg'
    return render_template('index.html', caption=caption, image_url=image_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    
    
    
    
    