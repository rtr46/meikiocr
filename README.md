# meikiocr

[![license: apache 2.0](https://img.shields.io/badge/license-apache%202.0-blue.svg)](https://github.com/your-github-username/meikiocr/blob/main/license)
[![detection model](https://img.shields.io/badge/hugging%20face-detection%20model-yellow)](https://huggingface.co/rtr46/meiki.text.detect.v0)
[![recognition model](https://img.shields.io/badge/hugging%20face-recognition%20model-yellow)](https://huggingface.co/rtr46/meiki.txt.recognition.v0)

high-speed, high-accuracy, local ocr for japanese video games.

`meikiocr` is a python-based ocr pipeline that combines state-of-the-art detection and recognition models to provide an unparalleled open-source solution for extracting japanese text from video games and similar rendered content.

| original image | ocr result |
| :---: | :---: |
| ![input](https://github.com/user-attachments/assets/646fb178-113c-4ad9-837a-d8b19e77261b) | ![input_ocrresult](https://github.com/user-attachments/assets/00d27896-5ebd-41fb-989a-7d259534fc92) |


```
ナルホド
こ、こんなにドキドキするの、
小学校の学級裁判のとき以来です。
```

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

the script is designed to be easy to run. first, clone the repository and install the required packages.

```bash
git clone https://github.com/your-github-username/meikiocr.git
cd meikiocr
pip install -r requirements.txt
```

### for nvidia gpu users (recommended)

for a massive performance boost, you can install the gpu-enabled version of onnx runtime. this will be detected automatically by the script.

```bash
# first, uninstall the cpu version if it's already installed
pip uninstall onnxruntime

# then, install the gpu version
pip install onnxruntime-gpu
```

## usage

run the pipeline by providing the path to an image.

```bash
python meiki_ocr.py examples/input.jpg
```

this will generate three output files in the same directory:
*   `input_ocrresult.jpg`: a copy of the input image with character bounding boxes drawn on it.
*   `input_ocrresult.json`: a structured file containing the recognized text and detailed information for each character, including its bounding box and confidence score.
*   `input_ocrresult.txt`: a plain text file with just the recognized lines.

### adjusting thresholds

you can adjust the confidence thresholds for both the text line detection and the character recognition models. lowering these values may help detect more text in challenging images.

```bash
# example with lower thresholds
python meiki_ocr.py examples/input.jpg --det_threshold 0.3 --rec_threshold 0.05
```

## how it works

`meikiocr` is a two-stage pipeline:
1.  **text detection:** the [meiki.text.detect.v0](https://huggingface.co/rtr46/meiki.text.detect.v0) model first identifies the bounding boxes of all horizontal text lines in the image.
2.  **text recognition:** each detected text line is then cropped and processed in a batch by the [meiki.text.recognition.v0](https://huggingface.co/rtr46/meiki.txt.recognition.v0) model, which recognizes the individual characters within it.

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
