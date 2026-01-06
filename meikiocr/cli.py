# ./meikiocr/cli.py

import argparse
import sys
import cv2
import json
import os
from .ocr import MeikiOCR
from . import __version__

def main():
    parser = argparse.ArgumentParser(
        description="High-speed, high-accuracy, local OCR for Japanese video games."
    )
    
    # Positional argument: Input image
    parser.add_argument("image_path", help="Path to the input image file.")
    
    # Optional arguments
    parser.add_argument("--output", "-o", help="Path to save the visualized image with bounding boxes.", default=None)
    parser.add_argument("--json", action="store_true", help="Output results in JSON format instead of plain text.")
    parser.add_argument("--det-threshold", type=float, default=0.5, help="Confidence threshold for text detection (default: 0.5).")
    parser.add_argument("--rec-threshold", type=float, default=0.1, help="Confidence threshold for character recognition (default: 0.1).")
    parser.add_argument("--punct-factor", type=float, default=1.0, help="Confidence factor for punctuation (default: 1.0).")
    parser.add_argument("--provider", type=str, default=None, help="ONNX Runtime provider (e.g., 'CUDAExecutionProvider', 'CPUExecutionProvider').")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    # Verify file existence
    if not os.path.exists(args.image_path):
        print(f"Error: File '{args.image_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # Load image
    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not decode image '{args.image_path}'.", file=sys.stderr)
        sys.exit(1)

    # Initialize OCR
    try:
        ocr = MeikiOCR(provider=args.provider)
    except Exception as e:
        print(f"Error initializing MeikiOCR: {e}", file=sys.stderr)
        sys.exit(1)

    # Run OCR
    results = ocr.run_ocr(
        image, 
        det_threshold=args.det_threshold, 
        rec_threshold=args.rec_threshold, 
        punct_conf_factor=args.punct_factor
    )

    # Handle Output
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        # Default behavior: Print recognized text lines
        valid_lines = [line['text'] for line in results if line['text']]
        if valid_lines:
            print('\n'.join(valid_lines))

    # Handle Visualization
    if args.output:
        vis_image = image.copy()
        for line in results:
            for char_info in line['chars']:
                x1, y1, x2, y2 = char_info['bbox']
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        cv2.imwrite(args.output, vis_image)
        # Only print status if not in JSON mode to keep stdout clean for piping
        if not args.json:
            print(f"\nSaved visualization to {args.output}")

if __name__ == "__main__":
    main()