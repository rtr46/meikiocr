# meikiocr

[![license: apache 2.0](https://img.shields.io/badge/license-apache%202.0-blue.svg)](https://github.com/your-github-username/meikiocr/blob/main/license)
[![hugging face space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-space-blue)](https://huggingface.co/spaces/rtr46/meikiocr)
[![detection model](https://img.shields.io/badge/hugging%20face-detection%20model-yellow)](https://huggingface.co/rtr46/meiki.text.detect.v0)
[![recognition model](https://img.shields.io/badge/hugging%20face-recognition%20model-yellow)](https://huggingface.co/rtr46/meiki.txt.recognition.v0)

high-speed, high-accuracy, local ocr for video games.

`meikiocr` is a python-based ocr pipeline that combines state-of-the-art detection and recognition models to provide an unparalleled open-source solution for extracting text from video games and similar rendered content. supports **japanese** (native, character-level detection) and **17+ other languages** via paddleocr onnx models.

| original image | ocr result |
| :---: | :---: |
| ![input](https://github.com/user-attachments/assets/646fb178-113c-4ad9-837a-d8b19e77261b) | ![input_ocrresult](https://github.com/user-attachments/assets/00d27896-5ebd-41fb-989a-7d259534fc92) |


```
ナルホド
こ、こんなにドキドキするの、
小学校の学級裁判のとき以来です。
```

---

## live demo

the easiest way to see `meikiocr` in action is to try the live demo hosted on hugging face spaces. no installation required!

**[try the meikiocr live demo here](https://huggingface.co/spaces/rtr46/meikiocr)**

---

## core features

*   **high accuracy:** purpose-built and trained on japanese video game text, `meikiocr` significantly outperforms general-purpose ocr tools like paddleocr or easyocr on this specific domain.
*   **high speed:** the architecture is pareto-optimal, delivering exceptional performance on both cpu and gpu.
*   **fully local & private:** unlike cloud-based services, `meikiocr` runs entirely on your machine, ensuring privacy and eliminating api costs or rate limits.
*   **cross-platform:** it works wherever onnx runtime runs, providing a much-needed local ocr solution for linux users.
*   **open & free:** both the code and the underlying models are freely available under permissive licenses.

## performance & benchmarks

`meikiocr` is built from two highly efficient models that establish a new pareto front for japanese text recognition. this means they offer a better accuracy/latency tradeoff than any other known open-weight model.

| detection (cpu) | detection (gpu) |
|:---:|:---:|
| ![accuracy_vs_cpu_latency](https://cdn-uploads.huggingface.co/production/uploads/68f7a26cfcf6939fd30fb19f/91aWIOgNQ9N8G7iaspRKX.png) | ![accuracy_vs_gpu_latency](https://cdn-uploads.huggingface.co/production/uploads/68f7a26cfcf6939fd30fb19f/61-T8E9RNnGtaHDCWcU23.png) |

| recognition (cpu) | recognition (gpu) |
| :---: | :---: |
| ![accuracy_vs_cpu_latency](https://cdn-uploads.huggingface.co/production/uploads/68f7a26cfcf6939fd30fb19f/NoTZVOLPhHMFW-O3fmgif.png) | ![accuracy_vs_gpu_latency](https://cdn-uploads.huggingface.co/production/uploads/68f7a26cfcf6939fd30fb19f/UQdnt0dN4qSpvBKLrkRZE.png) |

## installation

```bash
pip install meikiocr
```

### for nvidia gpu users (recommended)

for a massive performance boost, you can install the gpu-enabled version of the onnx runtime. this will be detected automatically by the script.

```bash
pip install meikiocr
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

## usage - cli

after installation, you can use the `meikiocr` tool directly from your terminal.

```bash
meikiocr image.png
```

### options

*   **language selection:** specify the recognition language (default: japanese).
    ```bash
    meikiocr image.png --language en    # english
    meikiocr image.png -l pt             # portuguese
    meikiocr image.png -l zh             # chinese
    ```
*   **save visualization:** draw bounding boxes and save the result to a file.
    ```bash
    meikiocr image.png --output result.jpg
    ```
*   **json output:** get detailed results (coordinates, confidence scores) for integration with other scripts.
    ```bash
    meikiocr image.png --json
    ```
*   **adjust thresholds:** fine-tune detection and recognition sensitivity.
    ```bash
    meikiocr image.png --det-threshold 0.6 --rec-threshold 0.2
    ```

run `meikiocr --help` for a full list of available options.


## usage - python

this is how meikiocr can be called. you can also run [demo.py](https://github.com/rtr46/meikiocr/blob/main/demo.py) for additional visual output.

```python
import cv2
import numpy as np
from urllib.request import urlopen
from meikiocr import MeikiOCR

IMAGE_URL = "https://huggingface.co/spaces/rtr46/meikiocr/resolve/main/example.jpg"

with urlopen(IMAGE_URL) as resp:
    image = cv2.imdecode(np.asarray(bytearray(resp.read()), dtype="uint8"), cv2.IMREAD_COLOR)

ocr = MeikiOCR()  # Japanese (default)
results = ocr.run_ocr(image)
print('\n'.join([line['text'] for line in results if line['text']]))

# For other languages:
ocr_en = MeikiOCR(language="en")  # English
ocr_pt = MeikiOCR(language="pt")  # Portuguese (uses Latin model)
ocr_zh = MeikiOCR(language="zh")  # Chinese
```

### adjusting thresholds

you can adjust the confidence thresholds for both the text line detection and the character recognition models. lowering the thresholds results in more detected text lines and characters, while higher values prevent false positives.

```python
MeikiOCR().run_ocr(self, image, det_threshold=0.8, rec_threshold=0.2) # less, but more confident text boxes and characters returned
```

### running dedicated detection

if you only care about the position of the text and not the content you can run the detection by itself, which is faster than running the whole ocr pipeline:
```python
MeikiOCR().run_detection(self, image, det_threshold=0.8, rec_threshold=0.2) # only returns text line coordinates (for horizontal and vertical text lines)
```
in the same way you can also run_recognition by itself on images of precropped (horizontal) text lines.

## how it works

`meikiocr` is a two-stage pipeline:
1.  **text detection:** the [meiki.text.detect.v0](https://huggingface.co/rtr46/meiki.text.detect.v0) model identifies bounding boxes of all horizontal text lines. this model is language-agnostic and works for any language.
2.  **text recognition:** each detected text line is cropped and processed by a recognition model:
    - **japanese:** uses [meiki.text.recognition.v0](https://huggingface.co/rtr46/meiki.txt.recognition.v0) with character-level detection
    - **other languages:** uses [paddleocr onnx models](https://huggingface.co/monkt/paddleocr-onnx) with ctc-based recognition

## supported languages

| code | language/script | model | char-level bboxes |
|------|-----------------|-------|-------------------|
| `ja` | japanese | native meikiocr | ✅ yes |
| `en` | english | paddleocr | ❌ no |
| `pt`, `es`, `fr`, `de`, `it`, `nl`, `pl`, `tr`, `latin` | latin script (32 langs) | paddleocr | ❌ no |
| `zh` | chinese | paddleocr | ❌ no |
| `ko` | korean | paddleocr | ❌ no |
| `ru`, `uk`, `cyrillic` | cyrillic | paddleocr | ❌ no |
| `th` | thai | paddleocr | ❌ no |
| `el` | greek | paddleocr | ❌ no |

## limitations

while `meikiocr` is state-of-the-art for its niche, it's important to understand its design constraints:
*   **domain specific:** it is highly optimized for rendered text from video games and may not perform well on handwritten or complex real-world scene text.
*   **horizontal text only:** it does not currently support vertical text.
*   **architectural limits:** the detection model is capped at finding 64 text boxes, and the recognition model can process up to 48 characters per line. these limits are sufficient for over 99% of video game scenarios but may be a constraint for other use cases.

## advanced usage & potential

the `meiki_ocr.py` script provides a straightforward implementation of a post-processing pipeline that selects the most confident prediction for each character. however, the raw output from the recognition model is richer and can be used for more advanced applications. for example, one could build a language-aware post-processing step using n-grams to correct ocr mistakes by considering alternative character predictions.

this opens the door for `meikiocr` to be integrated into a variety of projects.

## license

this project is licensed under the apache 2.0 license. see the [license](LICENSE) file for details.
