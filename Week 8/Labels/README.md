# Results 

I tested on 3 different VLMs for this attempt and the dataset used was Images1 consisting of 256 images.

## Aim

Prompted the models to detect simple and generic objects in each image and list them out.

Passed these labels of objects through a zero-shot object detection model to check if model is able to identify correctly.

## Accuracy

**Google's Gemma3-4b-it** : 56.1%

**Qwen2.5-3B-Instruct** : 39.0%

**LLava-v1.6-mistral-7b-hf** : 50.3%
