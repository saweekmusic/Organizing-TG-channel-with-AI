# SLife AI in Telegram

## Introduction
This is my step-by-step expiriment on how I have integrated AI into my everyday life.

One of the most important things is that for AI to be as private as possible, which means that all the models will be downloaded and running locally 24/7 in my house on a server or a home PC. The only thing that will have connection to the Internet is Telegram API to pull the messages and push the edits.

If you read this, you probably already know something about Telegram Bot API and know how to use it. If you don’t, I suggest for you to check the Telegram API Manual online at https://core.telegram.org/

This entire experiment and all the programs will be built using Python. So make sure you have Python 3.6+ installed, and in case you don’t have it you can either go to https://www.python.org/downloads/ or alternatively use use Homebrew on macOS:
```bash
brew install python
```

You can check the Python version with this command:
```bash
python3 --version
pip --version
```

## Installing AI models
Before making anything work we need to install AI models that we are going to use for our experiment. And then to check if all of them work as they should. After comparing different models, I have chosen the following:

### BERT (Text Model) by Facebook
*Reason: Excellent at understanding the overall context of text and can be fine-tuned for classification tasks.*

#### Installing
Hugging Face's Transformers and Torch (aka. PyTorch).

**Hugging Face's Transformers** — is a library that makes it super easy to grab and use powerful, pre-trained AI models like BART. Think of it as an app store for AI models that handle text, images, and more. We need it to access ready-made models for generating hashtags without building them from scratch. Hugging Face provides pre-trained BERT models that are easy to use.

**Torch (PyTorch)** — is the engine that actually runs these models on your computer. It handles all the heavy math and makes the models work smoothly, especially if you’re using a GPU. Hugging Face’s models use Torch under the hood, so we need it to make everything run.

Installation is as simple as it can be:
```bash
pip install transformers torch
```

> [!Tip]
> For better performance on macOS, you can use Apple’s Metal Performance Shaders if you’re running an M1 or M2 Mac. To install PyTorch optimized for Metal, use:
> ```bash
> pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/metal.html
> ```

#### Tesing
Run a quick test to check if BERT is installed properly:
```python
from transformers import pipeline

nlp = pipeline("text-classification", model = "bert-base-uncased")
print(nlp("Testing BERT installation!"))
```

If everything went well, after printing the last line, you will get something like:
```[{'label': 'LABEL_0', 'score': 0.6314278244972229}]``` and that is excactly what we need. Label is basically a tag and the score — is how sure BERT is about the input fitting to the lable. If you see something like that, it means you have successfully installed BERT. 

### EfficientNet (Image Model) ---- not
Reason: EfficientNet is good for mobile and small-scale applications, so it’s great for running on your laptop or PC.

#### Installing
```bash
pip install tensorflow
```

### Swin Transformers

### Whisper (Audio Model) by OpenAI


from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import requests
model_name = "google/efficientnet-b0"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
url = "https://www.andrewshoemaker.com/images/640/secret-glow-maui-beach-sunset.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
print(outputs)