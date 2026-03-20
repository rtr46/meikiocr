# ./meikiocr/ocr.py

import os
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import logging
import unicodedata

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- configuration ---
DET_MODEL_REPO = "rtr46/meiki.text.detect.v0"
DET_MODEL_NAME = "meiki.text.detect.v0.1.960x544.onnx"
REC_MODEL_REPO = "rtr46/meiki.txt.recognition.v0"
REC_MODEL_NAME = "meiki.text.rec.v0.960x32.onnx"
VREC_MODEL_NAME = "meiki.text.rec.v0.vertical.32x480.onnx"

INPUT_DET_WIDTH = 960
INPUT_DET_HEIGHT = 544

# Horizontal Recognition Dims
INPUT_REC_HEIGHT = 32
INPUT_REC_WIDTH = 960

# Vertical Recognition Dims
INPUT_VREC_WIDTH = 32
INPUT_VREC_HEIGHT = 480
VREC_MAX_CONTENT_HEIGHT = 420  # Height of segments when a split is forced
VREC_OVERLAP_PX = 64  # Overlap strictly by 64px in the scaled space

X_OVERLAP_THRESHOLD = 0.3
Y_OVERLAP_THRESHOLD = 0.3
EPSILON = 1e-6


def _get_model_path(repo_id, filename):
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        logger.error(f"Error downloading model {filename}: {e}")
        raise


class MeikiOCR:
    def __init__(self, provider=None, max_batch_size=8):
        """
        Initializes the meikiocr pipeline by loading detection and recognition models.

        Args:
            provider (str, optional): The ONNX Runtime execution provider to use. 
                                      Defaults to None, which lets ONNX Runtime choose.
                                      Recommended: 'CUDAExecutionProvider' for NVIDIA GPUs,
                                      'CPUExecutionProvider' for CPU.
            max_batch_size (int, optional): The maximum batch size for the recognition model
                                            to control memory usage. Defaults to 8.
        """
        ort.set_default_logger_severity(3)

        det_model_path = _get_model_path(DET_MODEL_REPO, DET_MODEL_NAME)
        rec_model_path = _get_model_path(REC_MODEL_REPO, REC_MODEL_NAME)
        vrec_model_path = _get_model_path(REC_MODEL_REPO, VREC_MODEL_NAME)

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
        self.vrec_session = ort.InferenceSession(vrec_model_path, providers=chosen_providers)

        self.active_provider = self.det_session.get_providers()[0]
        self.max_batch_size = max_batch_size
        logger.info(f"meikiocr initialized on: {self.active_provider}; max_batch_size = {self.max_batch_size}")

    def run_ocr(self, image, det_threshold=0.5, rec_threshold=0.1, punct_conf_factor=1.0):
        """
        Runs the full OCR pipeline on a given image.

        Args:
            image (np.ndarray): The input image in OpenCV format (BGR, HxWxC).
            det_threshold (float): Confidence threshold for text detection.
            rec_threshold (float): Confidence threshold for character recognition.
            punct_conf_factor (float): Confidence factor for punctuation characters.
                                       Values < 1.0 allow overlapping non-punctuation text to take precedence.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains the
                        recognized 'text' and a list of 'chars' with their bounding
                        boxes and confidence scores for a detected text line.
        """
        text_boxes = self.run_detection(image, det_threshold)
        logger.debug(f"Detection found {len(text_boxes)} text boxes.")

        if not text_boxes:
            return []

        results = [{'text': '', 'chars': [], 'orientation': ''} for _ in range(len(text_boxes))]

        h_indices = []
        v_indices = []
        for i, tb in enumerate(text_boxes):
            x1, y1, x2, y2 = tb['bbox']
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue

            if h > w:
                v_indices.append(i)
            else:
                h_indices.append(i)

        if h_indices:
            logger.debug(f"Processing {len(h_indices)} horizontal boxes.")
            self._process_recognition_pipeline(
                image, text_boxes, h_indices, results, rec_threshold, punct_conf_factor, 'horizontal'
            )

        if v_indices:
            logger.debug(f"Processing {len(v_indices)} vertical boxes.")
            self._process_recognition_pipeline(
                image, text_boxes, v_indices, results, rec_threshold, punct_conf_factor, 'vertical'
            )

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

        text_boxes = [{'bbox': [0, 0, img.shape[1], img.shape[0]]} for img in text_line_images]
        results = [{'text': '', 'chars': [], 'orientation': ''} for _ in range(len(text_line_images))]

        for i, image in enumerate(text_line_images):
            h, w = image.shape[:2]
            mode = 'vertical' if h > w else 'horizontal'

            rec_batch, valid_indices, crop_metadata = self._preprocess_for_recognition(
                image, [text_boxes[i]], [0], mode
            )
            if rec_batch is None:
                continue

            rec_raw = self._run_recognition_inference(rec_batch, mode)
            temp_results = [{'text': '', 'chars': [], 'orientation': ''}]
            self._postprocess_recognition_results(
                rec_raw, valid_indices, crop_metadata, conf_threshold, temp_results, punct_conf_factor, mode
            )
            results[i] = temp_results[0]

        return results

    # --- Internal methods ---

    def _preprocess_for_detection(self, image):
        h_orig, w_orig = image.shape[:2]
        scale = min(INPUT_DET_WIDTH / w_orig, INPUT_DET_HEIGHT / h_orig)
        w_resized, h_resized = int(w_orig * scale), int(h_orig * scale)
        resized = cv2.resize(image, (w_resized, h_resized), interpolation=cv2.INTER_LINEAR)
        normalized_resized = resized.astype(np.float32) / 255.0
        tensor = np.zeros((INPUT_DET_HEIGHT, INPUT_DET_WIDTH, 3), dtype=np.float32)
        tensor[:h_resized, :w_resized] = normalized_resized
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
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

    def _process_recognition_pipeline(self, image, text_boxes, indices, results, rec_threshold, punct_conf_factor, mode):
        rec_batch, valid_indices, crop_metadata = self._preprocess_for_recognition(image, text_boxes, indices, mode)

        if rec_batch is None:
            return

        all_labels_chunks, all_boxes_chunks, all_scores_chunks = [], [], []
        for i in range(0, len(rec_batch), self.max_batch_size):
            batch_chunk = rec_batch[i:i + self.max_batch_size]
            labels_chunk, boxes_chunk, scores_chunk = self._run_recognition_inference(batch_chunk, mode)
            all_labels_chunks.append(labels_chunk)
            all_boxes_chunks.append(boxes_chunk)
            all_scores_chunks.append(scores_chunk)

        all_rec_raw = (
            np.concatenate(all_labels_chunks, axis=0),
            np.concatenate(all_boxes_chunks, axis=0),
            np.concatenate(all_scores_chunks, axis=0)
        )
        self._postprocess_recognition_results(
            all_rec_raw, valid_indices, crop_metadata, rec_threshold, results, punct_conf_factor, mode
        )

    def _preprocess_for_recognition(self, image, text_boxes, indices, mode):
        tensors, valid_indices, crop_metadata = [], [], []

        for i in indices:
            tb = text_boxes[i]
            x1, y1, x2, y2 = tb['bbox']
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            h, w = crop.shape[:2]

            if mode == 'horizontal':
                new_h = INPUT_REC_HEIGHT
                scale = new_h / h
                new_w = int(round(w * scale))

                if new_w > INPUT_REC_WIDTH:
                    scale_w = INPUT_REC_WIDTH / new_w
                    new_w = INPUT_REC_WIDTH
                    new_h = int(round(new_h * scale_w))

                resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                pad_w, pad_h = INPUT_REC_WIDTH - new_w, INPUT_REC_HEIGHT - new_h
                padded = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)

                tensor = (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)
                tensors.append(tensor)
                valid_indices.append(i)
                crop_metadata.append({'orig_bbox': [x1, y1, x2, y2], 'effective_w': new_w, 'effective_h': new_h})

            else:  # mode == 'vertical'
                scale = INPUT_VREC_WIDTH / w
                h_scaled_full = h * scale

                logger.debug(
                    f"[Box {i}] Vertical original dims: h={h}, w={w}, scale={scale:.3f}, scaled_h={h_scaled_full:.1f}")

                # Only split if it absolutely exceeds the 480px model limit
                if h_scaled_full > INPUT_VREC_HEIGHT:
                    # When splitting, enforce the smaller VREC_MAX_CONTENT_HEIGHT to force padding
                    max_h_scaled = VREC_MAX_CONTENT_HEIGHT
                    segment_h_orig = VREC_MAX_CONTENT_HEIGHT / scale
                    stride_orig = (VREC_MAX_CONTENT_HEIGHT - VREC_OVERLAP_PX) / scale

                    y_starts = []
                    curr_y = y1
                    while curr_y + segment_h_orig < y2:
                        y_starts.append(curr_y)
                        curr_y += stride_orig

                    last_y = y2 - segment_h_orig
                    if not y_starts or last_y > y_starts[-1] + 1.0:
                        y_starts.append(last_y)
                else:
                    # If organically smaller than 480px (e.g. 478px), do not split. Process natively.
                    max_h_scaled = INPUT_VREC_HEIGHT
                    y_starts = [y1]
                    segment_h_orig = y2 - y1

                for seg_idx, sy1_f in enumerate(y_starts):
                    sy1 = int(round(sy1_f))
                    sy2 = int(round(sy1_f + segment_h_orig))
                    sy2 = min(sy2, y2)

                    segment_crop = image[sy1:sy2, x1:x2]
                    seg_h = segment_crop.shape[0]
                    if seg_h <= 0: continue

                    seg_new_h = min(int(round(seg_h * scale)), max_h_scaled)
                    resized = cv2.resize(segment_crop, (INPUT_VREC_WIDTH, seg_new_h), interpolation=cv2.INTER_LINEAR)

                    pad_h = INPUT_VREC_HEIGHT - seg_new_h
                    padded = np.pad(resized, ((0, pad_h), (0, 0), (0, 0)), constant_values=0)

                    tensor = (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)
                    tensors.append(tensor)
                    valid_indices.append(i)

                    logger.debug(
                        f"  -> Segment {seg_idx}: sy1={sy1}, sy2={sy2}, content_h={seg_new_h}, pad_bottom={pad_h}")

                    crop_metadata.append({
                        'orig_bbox': [x1, sy1, x2, sy2],
                        'effective_w': INPUT_VREC_WIDTH,
                        'effective_h': seg_new_h,
                        'segment_idx': seg_idx
                    })

        if not tensors: return None, [], []
        return np.stack(tensors, axis=0), valid_indices, crop_metadata

    def _run_recognition_inference(self, batch_tensor, mode):
        if batch_tensor is None: return []
        if mode == 'horizontal':
            orig_size = np.array([[INPUT_REC_WIDTH, INPUT_REC_HEIGHT]], dtype=np.int64)
            return self.rec_session.run(None, {"images": batch_tensor, "orig_target_sizes": orig_size})
        else:
            orig_size = np.array([[INPUT_VREC_WIDTH, INPUT_VREC_HEIGHT]], dtype=np.int64)
            return self.vrec_session.run(None, {"images": batch_tensor, "orig_target_sizes": orig_size})

    def _postprocess_recognition_results(self, raw_rec_outputs, valid_indices, crop_metadata, rec_conf_threshold,
                                         results, punct_conf_factor, mode):
        labels_batch, boxes_batch, scores_batch = raw_rec_outputs
        candidates_by_idx = {}

        for i, (labels, boxes, scores) in enumerate(zip(labels_batch, boxes_batch, scores_batch)):
            orig_idx = valid_indices[i]
            meta = crop_metadata[i]
            gx1, gy1, gx2, gy2 = meta['orig_bbox']
            crop_w, crop_h = gx2 - gx1, gy2 - gy1

            if orig_idx not in candidates_by_idx:
                candidates_by_idx[orig_idx] = []

            logger.debug(f"--- Processing Raw Results for Box {orig_idx} | Seg {meta.get('segment_idx', 0)} ---")

            for lbl, box, scr in zip(labels, boxes, scores):
                if scr < rec_conf_threshold:
                    continue

                char = chr(lbl)
                rx1, ry1, rx2, ry2 = box

                if mode == 'horizontal':
                    effective_w = meta['effective_w']
                    if rx1 >= effective_w:
                        continue

                    rx1, rx2 = min(rx1, effective_w), min(rx2, effective_w)
                    cx1, cx2 = (rx1 / effective_w) * crop_w, (rx2 / effective_w) * crop_w
                    cy1, cy2 = (ry1 / INPUT_REC_HEIGHT) * crop_h, (ry2 / INPUT_REC_HEIGHT) * crop_h

                    gx1_char, gy1_char = gx1 + int(cx1), gy1 + int(cy1)
                    gx2_char, gy2_char = gx1 + int(cx2), gy1 + int(cy2)

                    candidates_by_idx[orig_idx].append({
                        'char': char, 'bbox': [gx1_char, gy1_char, gx2_char, gy2_char],
                        'conf': float(scr), 'interval': (gx1_char, gx2_char)
                    })
                else:  # mode == 'vertical'
                    effective_h = meta['effective_h']

                    if ry1 >= effective_h:
                        continue

                    ry1, ry2 = min(ry1, effective_h), min(ry2, effective_h)

                    cx1, cx2 = (rx1 / INPUT_VREC_WIDTH) * crop_w, (rx2 / INPUT_VREC_WIDTH) * crop_w
                    cy1, cy2 = (ry1 / effective_h) * crop_h, (ry2 / effective_h) * crop_h

                    gx1_char, gy1_char = gx1 + int(cx1), gy1 + int(cy1)
                    gx2_char, gy2_char = gx1 + int(cx2), gy1 + int(cy2)

                    if gy2_char <= gy1_char:
                        continue

                    logger.debug(
                        f"  [KEPT]      '{char}' (conf: {float(scr):.2f}) mapping to global gy={gy1_char}-{gy2_char}")

                    candidates_by_idx[orig_idx].append({
                        'char': char, 'bbox': [gx1_char, gy1_char, gx2_char, gy2_char],
                        'conf': float(scr), 'interval': (gy1_char, gy2_char)
                    })

        overlap_threshold = X_OVERLAP_THRESHOLD if mode == 'horizontal' else Y_OVERLAP_THRESHOLD

        for orig_idx, candidates in candidates_by_idx.items():
            logger.debug(f"--- Running NMS on Combined Candidates for Box {orig_idx} ---")

            if punct_conf_factor != 1.0:
                for cand in candidates:
                    if unicodedata.category(cand['char']).startswith('P'):
                        cand['conf'] *= punct_conf_factor

            candidates.sort(key=lambda c: c['conf'], reverse=True)
            accepted = []
            accepted_intervals = []

            for cand in candidates:
                i1_c, i2_c = cand['interval']
                len_c = i2_c - i1_c + EPSILON
                is_overlap = False

                for i1_a, i2_a in accepted_intervals:
                    if (i1_c >= i2_a) or (i1_a >= i2_c):
                        continue

                    inter_start = max(i1_c, i1_a)
                    inter_end = min(i2_c, i2_a)
                    inter_len = max(0, inter_end - inter_start)

                    len_a = i2_a - i1_a + EPSILON
                    min_len = min(len_c, len_a)

                    if (inter_len / min_len) > overlap_threshold:
                        is_overlap = True
                        break

                if not is_overlap:
                    logger.debug(
                        f"  [NMS ACCEPT] '{cand['char']}' (conf: {cand['conf']:.2f}) at interval {cand['interval']}")
                    accepted.append(cand)
                    accepted_intervals.append(cand['interval'])
                else:
                    logger.debug(
                        f"  [NMS SUPPRESS] '{cand['char']}' (conf: {cand['conf']:.2f}) at interval {cand['interval']} due to overlap")

            accepted.sort(key=lambda c: c['interval'][0])

            result_chars = [{'char': c['char'], 'bbox': c['bbox'], 'conf': c['conf']} for c in accepted]
            text = ''.join(c['char'] for c in result_chars)

            logger.debug(f"--- FINAL TEXT BOX {orig_idx}: {text} ---")
            results[orig_idx] = {'text': text, 'chars': result_chars, 'orientation': mode}
