# Image Classifier

A simple deep learning app that uses a pre-trained ResNet-18 model from PyTorch to classify images into ImageNet categories. It uses Gradio for an interactive web interface.

[Visit Live Project](https://colab.research.google.com/github/mrishikreddy/Image-Classifier-RT8/blob/main/Image_classifier.ipynb)

---

## Table of Contents

- [Installation Instructions](#installation-instructions)  
- [Usage](#usage)  
- [Features](#features)

---

## Installation Instructions

### Prerequisites

- Python 3.x
- Google Colab or a local Jupyter environment (recommended for Gradio)
- Internet access (for model and label download)

### Steps to Run

1. Install Gradio:
   ```bash
   pip install gradio
   ```

2. Import and load the pre-trained ResNet-18 model:
   ```python
   import torch
   model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()
   ```

3. Define the prediction function using `torchvision` and `PIL`.

4. Launch the Gradio interface:
   ```python
   gr.Interface(fn=predict,
       inputs=gr.Image(type="pil"),
       outputs=gr.Label(num_top_classes=3),
       examples=["/content/lion.jpg", "/content/cheetah.jpg"]).launch()
   ```

---

## Usage

- Upload or select an image (e.g., lion, cheetah).
- The model processes the image and returns the top 3 predicted labels with confidence scores.
- Hosted temporarily via `gradio.live` (72-hour shareable link).

---

## Features

- Uses pre-trained ResNet-18 from PyTorch Hub.
- Predicts from 1,000 ImageNet categories.
- Interactive frontend built with Gradio.
- No training required – just plug and predict.
- Can be deployed easily to [Hugging Face Spaces](https://huggingface.co/spaces) with `gradio deploy`.

