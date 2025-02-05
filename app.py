from flask import Flask, render_template, request, send_file, jsonify
import os
from werkzeug.utils import secure_filename
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
from datetime import datetime
from PIL import Image
from pdf2image import convert_from_path
import shutil
# Add this if you're on Windows and have poppler installed in a custom location
import os
os.environ['PATH'] += r';C:\Program Files\poppler-24.08.0\Library\bin'  # Adjust this path to your poppler installation

app = Flask(__name__)

# Configure upload and download folders
UPLOAD_FOLDER = 'uploads'
EXTRACTED_FOLDER = 'extracted_tables'
DOWNLOAD_FOLDER = 'downloads'
ALLOWED_EXTENSIONS = {'pdf'}

# Create necessary directories
for folder in [UPLOAD_FOLDER, EXTRACTED_FOLDER, DOWNLOAD_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXTRACTED_FOLDER'] = EXTRACTED_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_tables(pdf_path):
    # Initialize the processor and model
    image_processor = AutoImageProcessor.from_pretrained(
        "microsoft/table-transformer-detection",
        use_fast=True
    )
    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection",
        ignore_mismatched_sizes=True
    )
    
    # Convert PDF to images - Add this line before the loop
    pdf_images = convert_from_path(pdf_path)  # Changed variable name to pdf_images
    extracted_tables = []
    
    # Process each page - Update the loop to use pdf_images
    for page_num, image in enumerate(pdf_images, start=1):  # Changed images to pdf_images
        # Process the image
        inputs = image_processor(
            images=image, 
            return_tensors="pt",
            size={
                "shortest_edge": 600,
                "longest_edge": 800
            }
        )
        outputs = model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.8, target_sizes=target_sizes)[0]
        
        # Process detected tables
        for idx, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
            if score >= 0.8:
                box = [int(i) for i in box.tolist()]
                xmin, ymin, xmax, ymax = box
                
                # Sort tables by their vertical position on the page
                y_position = ymin
                
                cropped_table = image.crop((xmin, ymin, xmax, ymax))
                cropped_table = cropped_table.convert('RGB')
                table_filename = f"table_page_{page_num}_idx_{idx}.jpg"
                save_path = os.path.join(app.config['EXTRACTED_FOLDER'], table_filename)
                cropped_table.save(save_path, 'JPEG', quality=95)
                extracted_tables.append({
                    'filename': table_filename,
                    'path': save_path,
                    'page': page_num,
                    'y_position': y_position
                })
    
    # Sort tables by page number and vertical position
    extracted_tables.sort(key=lambda x: (x['page'], x['y_position']))
    
    return extracted_tables

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Only clear the uploads and extracted_tables folders
        for folder in [UPLOAD_FOLDER, EXTRACTED_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error: {e}")
        
        # Save and process the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract tables
        extracted_tables = extract_tables(file_path)
        return jsonify({'tables': extracted_tables})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/extracted_tables/<filename>')
def serve_image(filename):
    return send_file(os.path.join(app.config['EXTRACTED_FOLDER'], filename))


@app.route('/download_selected', methods=['POST'])
@app.route('/download_selected', methods=['POST'])
def download_selected():
    selected_files = request.json.get('selected_files', [])
    
    # Create a new folder with timestamp in downloads directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    download_dir = os.path.join(app.config['DOWNLOAD_FOLDER'], timestamp)
    os.makedirs(download_dir, exist_ok=True)
    
    # Copy selected files to the new timestamped folder
    copied_files = []
    for filename in selected_files:
        src_path = os.path.join(app.config['EXTRACTED_FOLDER'], filename)
        dst_path = os.path.join(download_dir, filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_files.append(filename)
    
    return jsonify({
        'message': 'Files ready for download',
        'download_folder': timestamp,
        'files': copied_files
    })

if __name__ == '__main__':
    app.run(debug=True)