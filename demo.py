import cv2
import numpy as np
from urllib.request import urlopen
from meikiocr import MeikiOCR

# --- 1. Load Assets ---
IMAGE_URL = "https://huggingface.co/spaces/rtr46/meikiocr/resolve/main/example.jpg"

print(f"--- Downloading sample image from {IMAGE_URL} ---")
with urlopen(IMAGE_URL) as resp:
    image_bytes = np.asarray(bytearray(resp.read()), dtype="uint8") # Read image data into a buffer
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)              # Decode buffer into an OpenCV image

print("--- Initializing MeikiOCR (models will be downloaded on first run) ---")
ocr = MeikiOCR() # Initialize the OCR pipeline

# --- 2. Run OCR ---
print("--- Running OCR on the image ---")
results = ocr.run_ocr(image) # Run the full OCR pipeline

# --- 3. Display and Save Results ---
print("\n--- Recognized Text ---")
full_text = '\n'.join([line['text'] for line in results if line['text']])
print(full_text)
print("-----------------------")

vis_image = image.copy() # Create a copy of the image for drawing
for line in results:
    for char_info in line['chars']:
        x1, y1, x2, y2 = char_info['bbox'] # Get character bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 1) # Draw a green box

output_path = "demo_result.jpg"
cv2.imwrite(output_path, vis_image) # Save the image with boxes
print(f"\n--- Saved visualization with character boxes to {output_path} ---")