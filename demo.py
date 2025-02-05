from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
from PIL import Image
from pdf2image import convert_from_path # Import to convert PDF to image

file_path = '/content/drive/MyDrive/Colab Notebooks/AAFC_PDCAAS/1-s2.0-S0308814617312839-Lentils.pdf'
# Convert the PDF pages to a list of PIL Images
images = convert_from_path(file_path)

# Initialize the processor and model
image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

# Loop over all pages in the PDF
for page_num, image in enumerate(images):
    print(f"Processing page {page_num + 1}")

    # Process the image
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    target_sizes = torch.tensor([image.size[::-1]])  # Ensure image dimensions are in the correct order
    results = image_processor.post_process_object_detection(outputs, threshold=0.8, target_sizes=target_sizes)[0]

    # Loop through the detected tables and save or process the bounding boxes
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

        # Crop the detected table from the image based on bounding box
        xmin, ymin, xmax, ymax = map(int, box)
        cropped_table = image.crop((xmin, ymin, xmax, ymax))

        # Optionally, save the cropped table as an image
        cropped_table.save(f"extracted_table_page_{page_num + 1}.png")

        # You can also apply OCR to extract text from the table image if needed
