import os
import argparse
import cv2
import numpy as np
import onnxruntime as ort
import json
import time
from huggingface_hub import hf_hub_download

# --- configuration ---
# model repositories on hugging face hub
DET_MODEL_REPO = "rtr46/meiki.text.detect.v0"
DET_MODEL_NAME = "meiki.text.detect.v0.1.960x544.onnx"
REC_MODEL_REPO = "rtr46/meiki.txt.recognition.v0"
REC_MODEL_NAME = "meiki.text.rec.v0.960x32.onnx"

# model input shapes
INPUT_DET_WIDTH = 960
INPUT_DET_HEIGHT = 544
INPUT_REC_HEIGHT = 32
INPUT_REC_WIDTH = 960

# post-processing parameters
X_OVERLAP_THRESHOLD = 0.3  # max x-overlap ratio to keep lower-confidence char
EPSILON = 1e-6


def get_model_path(repo_id, filename):
    """downloads a model from the hugging face hub if not cached and returns the path."""
    print(f"loading model: {filename} from {repo_id}")
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return model_path
    except Exception as e:
        print(f"error downloading model {filename}: {e}")
        raise

def load_image(image_path):
    """loads an image using opencv."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"image not found: {image_path}")
    return image


def visualize_results(image, results, output_path):
    """draws character bounding boxes on the image and saves it."""
    vis_img = image.copy()
    color = (0, 255, 0)  # green
    thickness = 1

    for res in results:
        for char_info in res['chars']:
            x1, y1, x2, y2 = char_info['bbox']
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)

    cv2.imwrite(output_path, vis_img)
    print(f"--- saved visualization to {output_path} ---")


# --- detection pipeline ---
def preprocess_for_detection(image, target_w=INPUT_DET_WIDTH, target_h=INPUT_DET_HEIGHT):
    """resizes image for the detection model."""
    h_orig, w_orig = image.shape[:2]
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # hwc -> chw
    input_tensor = np.expand_dims(input_tensor, axis=0)   # add batch dim
    scale_x = w_orig / target_w
    scale_y = h_orig / target_h
    return input_tensor, scale_x, scale_y


def run_detection_inference(session, input_tensor, target_w=INPUT_DET_WIDTH, target_h=INPUT_DET_HEIGHT):
    """runs inference on the detection model."""
    input_names = [inp.name for inp in session.get_inputs()]
    inputs = {
        input_names[0]: input_tensor,
        input_names[1]: np.array([[target_w, target_h]], dtype=np.int64)
    }
    return session.run(None, inputs)


def postprocess_detection_results(raw_outputs, scale_x, scale_y, conf_threshold):
    """filters detection results by confidence and scales boxes to original image size."""
    _, boxes, scores = raw_outputs
    boxes = boxes[0]
    scores = scores[0]

    text_boxes = []
    for box, score in zip(boxes, scores):
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = box
        # scale box coordinates back to original image size
        x1_orig = int(x1 * scale_x)
        y1_orig = int(y1 * scale_y)
        x2_orig = int(x2 * scale_x)
        y2_orig = int(y2 * scale_y)

        text_boxes.append({'bbox': [x1_orig, y1_orig, x2_orig, y2_orig], 'conf': float(score)})

    # sort text boxes from top to bottom
    text_boxes.sort(key=lambda tb: tb['bbox'][1])
    return text_boxes


# --- recognition pipeline ---
def preprocess_for_recognition(image, text_boxes):
    """crops and preprocesses each text line for the recognition model using numpy."""
    tensors = []
    valid_indices = []
    crop_metadata = []

    for i, tb in enumerate(text_boxes):
        x1, y1, x2, y2 = tb['bbox']
        width, height = x2 - x1, y2 - y1

        if width < height or width == 0 or height == 0:
            continue

        crop = image[y1:y2, x1:x2]
        h, w = crop.shape[:2]

        new_h = INPUT_REC_HEIGHT
        new_w = int(round(w * (new_h / h)))

        if new_w > INPUT_REC_WIDTH:
            scale = INPUT_REC_WIDTH / new_w
            new_w = INPUT_REC_WIDTH
            new_h = int(round(new_h * scale))

        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = INPUT_REC_WIDTH - new_w
        pad_h = INPUT_REC_HEIGHT - new_h
        padded = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)

        # replicate torchvision.totensor() with numpy:
        # 1. scale to [0, 1]
        # 2. transpose from (h, w, c) to (c, h, w)
        tensor = (padded.astype(np.float32) / 255.0)
        tensor = np.transpose(tensor, (2, 0, 1))

        tensors.append(tensor)
        valid_indices.append(i)
        crop_metadata.append({'orig_bbox': [x1, y1, x2, y2], 'effective_w': new_w})

    if not tensors:
        return None, [], []

    batch_tensor = np.stack(tensors, axis=0) # create a single batch from the list of tensors
    return batch_tensor, valid_indices, crop_metadata


def run_recognition_inference(session, batch_tensor):
    """runs inference on the recognition model."""
    if batch_tensor is None:
        return []
    orig_size = np.array([[INPUT_REC_WIDTH, INPUT_REC_HEIGHT]], dtype=np.int64)
    outputs = session.run(None, {"images": batch_tensor, "orig_target_sizes": orig_size})
    return outputs


def postprocess_recognition_results(raw_rec_outputs, valid_indices, crop_metadata, rec_conf_threshold, num_total_boxes):
    """processes raw recognition output to produce final text and character boxes."""
    labels_batch, boxes_batch, scores_batch = raw_rec_outputs
    full_results = [{'text': '', 'chars': []} for _ in range(num_total_boxes)]

    for i, (labels, boxes, scores) in enumerate(zip(labels_batch, boxes_batch, scores_batch)):
        meta = crop_metadata[i]
        gx1, gy1, gx2, gy2 = meta['orig_bbox']
        crop_w, crop_h = gx2 - gx1, gy2 - gy1
        effective_w = meta['effective_w']

        candidates = []
        for lbl, box, scr in zip(labels, boxes, scores):
            if scr < rec_conf_threshold:
                continue
            char = chr(lbl)

            rx1, ry1, rx2, ry2 = box
            rx1, rx2 = min(rx1, effective_w), min(rx2, effective_w)

            # map from recognition space -> crop space -> global space
            cx1, cx2 = (rx1 / effective_w) * crop_w, (rx2 / effective_w) * crop_w
            cy1, cy2 = (ry1 / INPUT_REC_HEIGHT) * crop_h, (ry2 / INPUT_REC_HEIGHT) * crop_h

            gx1_char, gy1_char = gx1 + int(cx1), gy1 + int(cy1)
            gx2_char, gy2_char = gx1 + int(cx2), gy1 + int(cy2)

            candidates.append({
                'char': char,
                'bbox': [gx1_char, gy1_char, gx2_char, gy2_char],
                'conf': float(scr),
                'x_interval': (gx1_char, gx2_char)
            })

        candidates.sort(key=lambda c: c['conf'], reverse=True)

        accepted = []
        for cand in candidates:
            x1_c, x2_c = cand['x_interval']
            width_c = x2_c - x1_c + EPSILON
            is_overlap = any(
                (max(0, min(x2_c, x2_a) - max(x1_c, x1_a)) / width_c) > X_OVERLAP_THRESHOLD
                for x1_a, x2_a in (acc['x_interval'] for acc in accepted)
            )
            if not is_overlap:
                accepted.append(cand)

        accepted.sort(key=lambda c: c['x_interval'][0])
        text = ''.join(c['char'] for c in accepted)
        result_chars = [{'char': c['char'], 'bbox': c['bbox'], 'conf': c['conf']} for c in accepted]
        full_results[valid_indices[i]] = {'text': text, 'chars': result_chars}

    return full_results


# --- main ---
def main():
    parser = argparse.ArgumentParser(description='meiki ocr pipeline')
    parser.add_argument('image_path', help='path to the input image.')
    parser.add_argument('--det_threshold', type=float, default=0.5, help='confidence threshold for text detection.')
    parser.add_argument('--rec_threshold', type=float, default=0.1, help='confidence threshold for character recognition.')
    args = parser.parse_args()

    # --- 1. load assets (models are downloaded automatically) ---
    image = load_image(args.image_path)
    det_model_path = get_model_path(DET_MODEL_REPO, DET_MODEL_NAME)
    rec_model_path = get_model_path(REC_MODEL_REPO, REC_MODEL_NAME)

    ort.set_default_logger_severity(3) # suppress verbose onnx runtime logging
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    det_session = ort.InferenceSession(det_model_path, providers=providers)
    rec_session = ort.InferenceSession(rec_model_path, providers=providers)

    # print the execution provider being used by onnx runtime
    active_provider = det_session.get_providers()[0]
    print(f"--- ocr pipeline running on: {active_provider} ---")


    # --- 2. run detection ---
    det_input, sx, sy = preprocess_for_detection(image)
    det_raw = run_detection_inference(det_session, det_input)
    text_boxes = postprocess_detection_results(det_raw, sx, sy, args.det_threshold)
    print(f"found {len(text_boxes)} text boxes.")

    if not text_boxes:
        print("no text detected.")
        return

    # --- 3. run recognition ---
    rec_batch, valid_indices, crop_metadata = preprocess_for_recognition(image, text_boxes)
    rec_raw = run_recognition_inference(rec_session, rec_batch)
    results = postprocess_recognition_results(rec_raw, valid_indices, crop_metadata, args.rec_threshold, len(text_boxes))

    # --- 4. save output ---
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    output_json_path = f"{base_name}_ocrresult.json"
    output_txt_path = f"{base_name}_ocrresult.txt"
    output_img_path = f"{base_name}_ocrresult.jpg"

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"--- saved json result to {output_json_path} ---")

    full_text = '\n'.join(res['text'] for res in results if res['text'])
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    print(f"--- saved text result to {output_txt_path} ---")

    visualize_results(image, results, output_img_path)

if __name__ == '__main__':
    main()