# ./meikiocr/ocr.py

import os
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import logging
import unicodedata

from .ctc_decoder import load_dictionary, ctc_batch_decode

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- Meiki native model configuration ---
DET_MODEL_REPO = "rtr46/meiki.text.detect.v0"
DET_MODEL_NAME = "meiki.text.detect.v0.1.960x544.onnx"
REC_MODEL_REPO = "rtr46/meiki.txt.recognition.v0"
REC_MODEL_NAME = "meiki.text.rec.v0.960x32.onnx"

# --- PaddleOCR ONNX model configuration ---
PADDLEOCR_REPO = "monkt/paddleocr-onnx"

# Language to model path mapping
# "meiki" type uses native meikiocr recognition (optimized for Japanese games)
# "paddle" type uses PaddleOCR ONNX models
LANGUAGE_MODELS = {
    # Japanese - use native meikiocr (best for video games)
    "ja": {"type": "meiki"},
    
    # English - dedicated English model
    "en": {"type": "paddle", "path": "languages/english"},
    
    # Latin script languages (32 languages including Portuguese, Spanish, French, etc.)
    "pt": {"type": "paddle", "path": "languages/latin"},
    "es": {"type": "paddle", "path": "languages/latin"},
    "fr": {"type": "paddle", "path": "languages/latin"},
    "de": {"type": "paddle", "path": "languages/latin"},
    "it": {"type": "paddle", "path": "languages/latin"},
    "nl": {"type": "paddle", "path": "languages/latin"},
    "pl": {"type": "paddle", "path": "languages/latin"},
    "tr": {"type": "paddle", "path": "languages/latin"},
    "latin": {"type": "paddle", "path": "languages/latin"},
    
    # East Asian
    "zh": {"type": "paddle", "path": "languages/chinese"},
    "ko": {"type": "paddle", "path": "languages/korean"},
    
    # Cyrillic (Russian, Ukrainian, etc.)
    "ru": {"type": "paddle", "path": "languages/eslav"},
    "uk": {"type": "paddle", "path": "languages/eslav"},
    "cyrillic": {"type": "paddle", "path": "languages/eslav"},
    
    # Other scripts
    "th": {"type": "paddle", "path": "languages/thai"},
    "el": {"type": "paddle", "path": "languages/greek"},
}

INPUT_DET_WIDTH = 960
INPUT_DET_HEIGHT = 544
INPUT_REC_HEIGHT = 32
INPUT_REC_WIDTH = 960
PADDLE_REC_HEIGHT = 48  # PaddleOCR v5 uses 48px height

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
    def __init__(self, language="ja", provider=None, max_batch_size=8):
        """
        Initializes the meikiocr pipeline by loading detection and recognition models.

        Args:
            language (str, optional): Recognition language code. Defaults to "ja" (Japanese).
                                      Supported: ja, en, pt, es, fr, de, it, nl, pl, tr, latin,
                                      zh, ko, ru, uk, cyrillic, th, el.
                                      Japanese uses the optimized meikiocr recognition model.
                                      Other languages use PaddleOCR ONNX models.
            provider (str, optional): The ONNX Runtime execution provider to use. 
                                      Defaults to None, which lets ONNX Runtime choose.
                                      Recommended: 'CUDAExecutionProvider' for NVIDIA GPUs,
                                      'CPUExecutionProvider' for CPU.
            max_batch_size (int, optional): The maximum batch size for the recognition model
                                            to control memory usage. Defaults to 8.
        """
        ort.set_default_logger_severity(3)
        
        # Validate language
        if language not in LANGUAGE_MODELS:
            available = ", ".join(sorted(LANGUAGE_MODELS.keys()))
            raise ValueError(f"Unsupported language '{language}'. Available: {available}")
        
        self.language = language
        self.lang_config = LANGUAGE_MODELS[language]
        self.use_paddle_rec = self.lang_config["type"] == "paddle"
        
        # Determine execution providers
        available_providers = ort.get_available_providers()
        if provider and provider in available_providers:
            chosen_providers = [provider]
        elif 'CUDAExecutionProvider' in available_providers:
            chosen_providers = ['CUDAExecutionProvider']
        elif 'CPUExecutionProvider' in available_providers:
            chosen_providers = ['CPUExecutionProvider']
        else:
            chosen_providers = available_providers
        
        # Load detection model (always meikiocr - language agnostic)
        det_model_path = _get_model_path(DET_MODEL_REPO, DET_MODEL_NAME)
        self.det_session = ort.InferenceSession(det_model_path, providers=chosen_providers)
        
        # Load recognition model based on language
        if self.use_paddle_rec:
            # PaddleOCR ONNX recognition
            paddle_path = self.lang_config["path"]
            rec_model_path = _get_model_path(PADDLEOCR_REPO, f"{paddle_path}/rec.onnx")
            dict_path = _get_model_path(PADDLEOCR_REPO, f"{paddle_path}/dict.txt")
            self.rec_session = ort.InferenceSession(rec_model_path, providers=chosen_providers)
            self.dictionary = load_dictionary(dict_path)
            logger.info(f"Loaded PaddleOCR recognition for '{language}' ({len(self.dictionary)} chars)")
        else:
            # Native meikiocr recognition (Japanese)
            rec_model_path = _get_model_path(REC_MODEL_REPO, REC_MODEL_NAME)
            self.rec_session = ort.InferenceSession(rec_model_path, providers=chosen_providers)
            self.dictionary = None
            logger.info(f"Loaded native meikiocr recognition for '{language}'")
        
        self.active_provider = self.det_session.get_providers()[0]
        self.max_batch_size = max_batch_size
        logger.info(f"meikiocr running on: {self.active_provider}; max_batch_size = {self.max_batch_size}")

    def run_ocr(self, image, det_threshold=0.5, rec_threshold=0.1, punct_conf_factor=1.0):
        """
        Runs the full OCR pipeline on a given image.

        Args:
            image (np.ndarray): The input image in OpenCV format (BGR, HxWxC).
            det_threshold (float): Confidence threshold for text detection.
            rec_threshold (float): Confidence threshold for character recognition.
                                   (Only used for native meikiocr Japanese recognition)
            punct_conf_factor (float): Confidence factor for punctuation characters.
                                       (Only used for native meikiocr Japanese recognition)

        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains the
                        recognized 'text' and a list of 'chars' with their bounding
                        boxes and confidence scores for a detected text line.
                        Note: For non-Japanese languages, 'chars' contains text split
                        by character without individual bounding boxes.
        """
        text_boxes = self.run_detection(image, det_threshold)
        
        if not text_boxes:
            return []

        if self.use_paddle_rec:
            # PaddleOCR recognition path
            return self._run_paddle_ocr_pipeline(image, text_boxes)
        else:
            # Native meikiocr recognition path (Japanese)
            return self._run_meiki_ocr_pipeline(image, text_boxes, rec_threshold, punct_conf_factor)
    
    def _run_meiki_ocr_pipeline(self, image, text_boxes, rec_threshold, punct_conf_factor):
        """Run native meikiocr recognition for Japanese text."""
        rec_batch, valid_indices, crop_metadata = self._preprocess_for_recognition(image, text_boxes)
        
        if rec_batch is None:
            return [{'text': '', 'chars': []} for _ in range(len(text_boxes))]

        # Process the recognition in smaller batches to control memory usage
        all_labels_chunks, all_boxes_chunks, all_scores_chunks = [], [], []
        for i in range(0, len(rec_batch), self.max_batch_size):
            batch_chunk = rec_batch[i:i + self.max_batch_size]
            labels_chunk, boxes_chunk, scores_chunk = self._run_recognition_inference(batch_chunk)
            all_labels_chunks.append(labels_chunk)
            all_boxes_chunks.append(boxes_chunk)
            all_scores_chunks.append(scores_chunk)

        all_rec_raw = (
            np.concatenate(all_labels_chunks, axis=0),
            np.concatenate(all_boxes_chunks, axis=0),
            np.concatenate(all_scores_chunks, axis=0)
        )
        results = self._postprocess_recognition_results(
            all_rec_raw,
            valid_indices,
            crop_metadata,
            rec_threshold,
            len(text_boxes),
            punct_conf_factor
        )
        return results
    
    def _run_paddle_ocr_pipeline(self, image, text_boxes):
        """Run PaddleOCR ONNX recognition for non-Japanese languages."""
        rec_batch, valid_indices, crop_metadata = self._preprocess_for_paddle_recognition(image, text_boxes)
        
        if rec_batch is None:
            return [{'text': '', 'chars': []} for _ in range(len(text_boxes))]
        
        # Process recognition in batches
        all_logits = []
        for i in range(0, len(rec_batch), self.max_batch_size):
            batch_chunk = rec_batch[i:i + self.max_batch_size]
            logits = self._run_paddle_recognition_inference(batch_chunk)
            all_logits.append(logits)
        
        all_logits = np.concatenate(all_logits, axis=0)
        
        # CTC decode
        texts = ctc_batch_decode(all_logits, self.dictionary)
        
        # Build results
        full_results = [{'text': '', 'chars': []} for _ in range(len(text_boxes))]
        for i, text in enumerate(texts):
            orig_idx = valid_indices[i]
            bbox = crop_metadata[i]['orig_bbox']
            # Create chars list from text (without individual bboxes for PaddleOCR)
            chars = [{'char': c, 'bbox': bbox, 'conf': 1.0} for c in text]
            full_results[orig_idx] = {'text': text, 'chars': chars}
        
        return full_results

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

    def run_recognition(self, text_line_images, conf_threshold=0.1, punct_conf_factor=1.0):
        """
        Runs only the text recognition part of the pipeline on a batch of text line images.
        Note: This is an advanced method. `run_ocr` is recommended for general use.
        
        Args:
            text_line_images (list[np.ndarray]): A list of cropped text line images (BGR, HxWxC).
            conf_threshold (float): Confidence threshold for character recognition.
            punct_conf_factor (float): Confidence factor for punctuation characters.
                                       Values < 1.0 allow overlapping non-punctuation text to take precedence.

        Returns:
            list[dict]: A list of recognition results, one for each input image.
        """
        if not text_line_images:
            return []

        # Create dummy text_boxes to fit the existing pipeline.
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
            result = self._postprocess_recognition_results(
                rec_raw,
                valid_indices,
                crop_metadata,
                conf_threshold,
                1,
                punct_conf_factor
            )
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

    def _postprocess_recognition_results(self, raw_rec_outputs, valid_indices, crop_metadata, rec_conf_threshold,
                                         num_total_boxes, punct_conf_factor):
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
                if rx1 >= effective_w:
                    continue
                rx1, rx2 = min(rx1, effective_w), min(rx2, effective_w)
                
                cx1, cx2 = (rx1 / effective_w) * crop_w, (rx2 / effective_w) * crop_w
                cy1, cy2 = (ry1 / INPUT_REC_HEIGHT) * crop_h, (ry2 / INPUT_REC_HEIGHT) * crop_h
                
                gx1_char, gy1_char = gx1 + int(cx1), gy1 + int(cy1)
                gx2_char, gy2_char = gx1 + int(cx2), gy1 + int(cy2)
                
                candidates.append({
                    'char': char, 'bbox': [gx1_char, gy1_char, gx2_char, gy2_char],
                    'conf': float(scr), 'x_interval': (gx1_char, gx2_char)
                })

            if punct_conf_factor != 1.0:
                for cand in candidates:
                    if unicodedata.category(cand['char']).startswith('P'):
                        cand['conf'] *= punct_conf_factor

            candidates.sort(key=lambda c: c['conf'], reverse=True)
            accepted = []
            accepted_intervals = []
            for cand in candidates:
                x1_c, x2_c = cand['x_interval']
                width_c = x2_c - x1_c + EPSILON
                is_overlap = False

                for x1_a, x2_a in accepted_intervals:
                    if (x1_c >= x2_a) or (x1_a >= x2_c):
                        continue
                    if ((min(x2_c, x2_a) - max(x1_c, x1_a)) / width_c) > X_OVERLAP_THRESHOLD:
                        is_overlap = True
                        break

                if not is_overlap:
                    accepted.append(cand)
                    accepted_intervals.append(cand['x_interval'])

            accepted.sort(key=lambda c: c['x_interval'][0])
            text = ''.join(c['char'] for c in accepted)
            result_chars = [{'char': c['char'], 'bbox': c['bbox'], 'conf': c['conf']} for c in accepted]
            full_results[valid_indices[i]] = {'text': text, 'chars': result_chars}
            
        return full_results

    # --- PaddleOCR-specific internal methods ---
    
    def _preprocess_for_paddle_recognition(self, image, text_boxes):
        """
        Preprocess text line crops for PaddleOCR ONNX recognition.
        PaddleOCR expects: (batch, 3, 48, width) with dynamic width.
        """
        tensors, valid_indices, crop_metadata = [], [], []
        max_width = 0
        crops_info = []
        
        for i, tb in enumerate(text_boxes):
            x1, y1, x2, y2 = tb['bbox']
            width, height = x2 - x1, y2 - y1
            if width < height or width <= 0 or height <= 0:
                continue
            
            crop = image[y1:y2, x1:x2]
            h, w = crop.shape[:2]
            
            # Resize to target height while maintaining aspect ratio
            new_h = PADDLE_REC_HEIGHT
            new_w = int(round(w * (PADDLE_REC_HEIGHT / h)))
            
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            crops_info.append({
                'resized': resized,
                'new_w': new_w,
                'orig_idx': i,
                'orig_bbox': [x1, y1, x2, y2]
            })
            max_width = max(max_width, new_w)
        
        if not crops_info:
            return None, [], []
        
        # Pad all crops to max width and normalize
        for info in crops_info:
            resized = info['resized']
            new_w = info['new_w']
            
            # Pad to max width
            pad_w = max_width - new_w
            if pad_w > 0:
                padded = np.pad(resized, ((0, 0), (0, pad_w), (0, 0)), constant_values=0)
            else:
                padded = resized
            
            # Normalize: PaddleOCR uses (img / 255.0 - 0.5) / 0.5 = img / 127.5 - 1.0
            tensor = (padded.astype(np.float32) / 127.5) - 1.0
            tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW
            tensors.append(tensor)
            valid_indices.append(info['orig_idx'])
            crop_metadata.append({'orig_bbox': info['orig_bbox'], 'effective_w': new_w})
        
        return np.stack(tensors, axis=0), valid_indices, crop_metadata
    
    def _run_paddle_recognition_inference(self, batch_tensor):
        """
        Run PaddleOCR ONNX recognition inference.
        Returns softmax probabilities for CTC decoding.
        """
        if batch_tensor is None:
            return np.array([])
        
        # PaddleOCR rec model expects "x" as input name
        input_name = self.rec_session.get_inputs()[0].name
        outputs = self.rec_session.run(None, {input_name: batch_tensor})
        
        # Output is typically shaped (batch, seq_len, num_classes)
        return outputs[0]