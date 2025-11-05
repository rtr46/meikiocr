# ./meikiocr/ocr.py

import os
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
import onnxruntime as ort

# --- configuration ---
DET_MODEL_REPO = "rtr46/meiki.text.detect.v0"
DET_MODEL_NAME = "meiki.text.detect.v0.1.960x544.onnx"
REC_MODEL_REPO = "rtr46/meiki.txt.recognition.v0"
REC_MODEL_NAME = "meiki.text.rec.v0.960x32.onnx"

INPUT_DET_WIDTH = 960
INPUT_DET_HEIGHT = 544
INPUT_REC_HEIGHT = 32
INPUT_REC_WIDTH = 960

X_OVERLAP_THRESHOLD = 0.3
EPSILON = 1e-6

def _get_model_path(repo_id, filename):
    """Downloads a model from the hugging face hub if not cached and returns the path."""
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        print(f"Error downloading model {filename}: {e}")
        raise

class MeikiOCR:
    def __init__(self, provider=None):
        """
        Initializes the meikiocr pipeline by loading detection and recognition models.

        Args:
            provider (str, optional): The ONNX Runtime execution provider to use. 
                                      Defaults to None, which lets ONNX Runtime choose.
                                      Recommended: 'CUDAExecutionProvider' for NVIDIA GPUs,
                                      'CPUExecutionProvider' for CPU.
        """
        ort.set_default_logger_severity(3)
        
        det_model_path = _get_model_path(DET_MODEL_REPO, DET_MODEL_NAME)
        rec_model_path = _get_model_path(REC_MODEL_REPO, REC_MODEL_NAME)

        available_providers = ort.get_available_providers()
        if provider and provider in available_providers:
            chosen_providers = [provider]
        elif 'CUDAExecutionProvider' in available_providers:
            chosen_providers = ['CUDAExecutionProvider']
        elif 'CPUExecutionProvider' in available_providers:
            chosen_providers = ['CPUExecutionProvider']
        else:
            chosen_providers = available_providers
        
        self.det_session = ort.InferenceSession(det_model_path, providers=chosen_providers)
        self.rec_session = ort.InferenceSession(rec_model_path, providers=chosen_providers)
        
        self.active_provider = self.det_session.get_providers()[0]
        print(f"--- meikiocr running on: {self.active_provider} ---")

    def run_ocr(self, image, det_threshold=0.5, rec_threshold=0.1):
        """
        Runs the full OCR pipeline on a given image.

        Args:
            image (np.ndarray): The input image in OpenCV format (BGR, HxWxC).
            det_threshold (float): Confidence threshold for text detection.
            rec_threshold (float): Confidence threshold for character recognition.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains the
                        recognized 'text' and a list of 'chars' with their bounding
                        boxes and confidence scores for a detected text line.
        """
        text_boxes = self.run_detection(image, det_threshold)
        
        if not text_boxes:
            return []

        rec_batch, valid_indices, crop_metadata = self._preprocess_for_recognition(image, text_boxes)
        
        if rec_batch is None:
            return [{'text': '', 'chars': []} for _ in range(len(text_boxes))]

        rec_raw = self._run_recognition_inference(rec_batch)
        results = self._postprocess_recognition_results(rec_raw, valid_indices, crop_metadata, rec_threshold, len(text_boxes))
        
        return results

    def run_detection(self, image, conf_threshold=0.5):
        """
        Runs only the text detection part of the pipeline.

        Args:
            image (np.ndarray): The input image in OpenCV format (BGR, HxWxC).
            conf_threshold (float): Confidence threshold for text detection.

        Returns:
            list[dict]: A list of detected text boxes, sorted from top to bottom.
                        Each box is a dictionary with 'bbox' and 'conf'.
        """
        det_input, scale = self._preprocess_for_detection(image)
        det_raw = self._run_detection_inference(det_input, scale)
        text_boxes = self._postprocess_detection_results(det_raw, image, conf_threshold)
        return text_boxes

    def run_recognition(self, text_line_images, conf_threshold=0.1):
        """
        Runs only the text recognition part of the pipeline on a batch of text line images.
        Note: This is an advanced method. `run_ocr` is recommended for general use.
        
        Args:
            text_line_images (list[np.ndarray]): A list of cropped text line images (BGR, HxWxC).
            conf_threshold (float): Confidence threshold for character recognition.

        Returns:
            list[dict]: A list of recognition results, one for each input image.
        """
        # This method requires creating dummy text_boxes to fit the existing pipeline
        text_boxes = [{'bbox': [0, 0, img.shape[1], img.shape[0]]} for img in text_line_images]
        
        # We need to process each image as if it were a crop from a larger canvas.
        # For simplicity, we process them one by one, though batching is possible with more complex metadata handling.
        results = []
        for i, image in enumerate(text_line_images):
            rec_batch, valid_indices, crop_metadata = self._preprocess_for_recognition(image, [text_boxes[i]])
            if rec_batch is None:
                results.append({'text': '', 'chars': []})
                continue
            rec_raw = self._run_recognition_inference(rec_batch)
            result = self._postprocess_recognition_results(rec_raw, valid_indices, crop_metadata, conf_threshold, 1)
            results.extend(result)
            
        return results

    # --- Internal "private" methods (prefixed with _) ---
    
    def _preprocess_for_detection(self, image):
        h_orig, w_orig = image.shape[:2]
        scale = min(INPUT_DET_WIDTH / w_orig, INPUT_DET_HEIGHT / h_orig)
        w_resized, h_resized = int(w_orig * scale), int(h_orig * scale)
        resized = cv2.resize(image, (w_resized, h_resized), interpolation=cv2.INTER_LINEAR)
        normalized_resized = resized.astype(np.float32) / 255.0
        tensor = np.zeros((INPUT_DET_HEIGHT, INPUT_DET_WIDTH, 3), dtype=np.float32)
        tensor[:h_resized, :w_resized] = normalized_resized
        tensor = np.transpose(tensor, (2, 0, 1)) # HWC -> CHW
        tensor = np.expand_dims(tensor, axis=0)  # Add batch dimension
        return tensor, scale

    def _run_detection_inference(self, tensor: np.ndarray, scale):
        inputs = {
            self.det_session.get_inputs()[0].name: tensor,
            self.det_session.get_inputs()[1].name: np.array([[INPUT_DET_WIDTH / scale, INPUT_DET_HEIGHT / scale]], dtype=np.int64)
        }
        return self.det_session.run(None, inputs)

    def _postprocess_detection_results(self, raw_outputs: list, image, conf_threshold: float):
        h_orig, w_orig = image.shape[:2]
        _, boxes, scores = raw_outputs
        boxes, scores = boxes[0], scores[0]
        confident_boxes = boxes[scores > conf_threshold]
        if confident_boxes.shape[0] == 0:
            return []
        max_bounds = np.array([w_orig, h_orig, w_orig, h_orig])
        clamped_boxes = np.clip(confident_boxes, 0, max_bounds).astype(np.int32)
        text_boxes = [{'bbox': box.tolist()} for box in clamped_boxes]
        text_boxes.sort(key=lambda tb: tb['bbox'][1])
        return text_boxes

    def _preprocess_for_recognition(self, image, text_boxes):
        tensors, valid_indices, crop_metadata = [], [], []
        for i, tb in enumerate(text_boxes):
            x1, y1, x2, y2 = tb['bbox']
            width, height = x2 - x1, y2 - y1
            if width < height or width <= 0 or height <= 0:
                continue
            
            crop = image[y1:y2, x1:x2]
            h, w = crop.shape[:2]
            new_h, new_w = INPUT_REC_HEIGHT, int(round(w * (INPUT_REC_HEIGHT / h)))
            if new_w > INPUT_REC_WIDTH:
                scale = INPUT_REC_WIDTH / new_w
                new_w, new_h = INPUT_REC_WIDTH, int(round(new_h * scale))
            
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pad_w, pad_h = INPUT_REC_WIDTH - new_w, INPUT_REC_HEIGHT - new_h
            padded = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
            
            tensor = (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)
            tensors.append(tensor)
            valid_indices.append(i)
            crop_metadata.append({'orig_bbox': [x1, y1, x2, y2], 'effective_w': new_w})

        if not tensors: return None, [], []
        return np.stack(tensors, axis=0), valid_indices, crop_metadata

    def _run_recognition_inference(self, batch_tensor):
        if batch_tensor is None: return []
        orig_size = np.array([[INPUT_REC_WIDTH, INPUT_REC_HEIGHT]], dtype=np.int64)
        return self.rec_session.run(None, {"images": batch_tensor, "orig_target_sizes": orig_size})

    def _postprocess_recognition_results(self, raw_rec_outputs, valid_indices, crop_metadata, rec_conf_threshold, num_total_boxes):
        labels_batch, boxes_batch, scores_batch = raw_rec_outputs
        full_results = [{'text': '', 'chars': []} for _ in range(num_total_boxes)]

        for i, (labels, boxes, scores) in enumerate(zip(labels_batch, boxes_batch, scores_batch)):
            meta = crop_metadata[i]
            gx1, gy1, gx2, gy2 = meta['orig_bbox']
            crop_w, crop_h = gx2 - gx1, gy2 - gy1
            effective_w = meta['effective_w']
            
            candidates = []
            for lbl, box, scr in zip(labels, boxes, scores):
                if scr < rec_conf_threshold: continue
                char = chr(lbl)
                rx1, ry1, rx2, ry2 = box
                rx1, rx2 = min(rx1, effective_w), min(rx2, effective_w)
                
                cx1, cx2 = (rx1 / effective_w) * crop_w, (rx2 / effective_w) * crop_w
                cy1, cy2 = (ry1 / INPUT_REC_HEIGHT) * crop_h, (ry2 / INPUT_REC_HEIGHT) * crop_h
                
                gx1_char, gy1_char = gx1 + int(cx1), gy1 + int(cy1)
                gx2_char, gy2_char = gx1 + int(cx2), gy1 + int(cy2)
                
                candidates.append({
                    'char': char, 'bbox': [gx1_char, gy1_char, gx2_char, gy2_char],
                    'conf': float(scr), 'x_interval': (gx1_char, gx2_char)
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