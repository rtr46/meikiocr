# ./meikiocr/ctc_decoder.py

"""
CTC (Connectionist Temporal Classification) decoder for PaddleOCR ONNX models.
Implements greedy decoding for sequence-to-sequence text recognition.
"""

import numpy as np


def load_dictionary(dict_path: str) -> list[str]:
    """
    Load character dictionary from file.
    
    Args:
        dict_path: Path to dictionary text file (one character per line)
        
    Returns:
        List of characters, with blank token at index 0
    """
    with open(dict_path, 'r', encoding='utf-8') as f:
        chars = [line.strip('\n') for line in f]
    # PaddleOCR uses blank at index 0, then dictionary chars, then space at end
    return [''] + chars + [' ']


def ctc_greedy_decode(logits: np.ndarray, dictionary: list[str]) -> str:
    """
    Decode CTC logits to text using greedy search.
    
    Args:
        logits: Shape (sequence_length, num_classes) - softmax outputs
        dictionary: Character vocabulary (index 0 is blank)
        
    Returns:
        Decoded text string
    """
    # Get best class at each timestep
    indices = np.argmax(logits, axis=-1)
    
    # Remove consecutive duplicates and blanks
    decoded_chars = []
    prev_idx = -1
    
    for idx in indices:
        # Skip if same as previous (CTC collapse) or if blank (index 0)
        if idx != prev_idx and idx != 0:
            if idx < len(dictionary):
                decoded_chars.append(dictionary[idx])
        prev_idx = idx
    
    return ''.join(decoded_chars)


def ctc_batch_decode(batch_logits: np.ndarray, dictionary: list[str]) -> list[str]:
    """
    Decode a batch of CTC logits.
    
    Args:
        batch_logits: Shape (batch, sequence_length, num_classes)
        dictionary: Character vocabulary
        
    Returns:
        List of decoded text strings
    """
    return [ctc_greedy_decode(logits, dictionary) for logits in batch_logits]
