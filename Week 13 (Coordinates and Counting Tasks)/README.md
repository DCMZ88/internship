# Data Curation 

## Coordinates and Counting Tasks

**Aim** : Obtain the output of detected objects, their respective bounding boxes and object count from VLMs. 

**Methodology**

1. Prompt the model to detect objects given an input image and output the labels and its respective bounding boxes respectively using structured text generation.
2. Save the outputs in a JSON file .

Vision Language Models:
  - Gemma3-4b-it
  - Qwen2.5-VL-7B-Instruct
  - Florence-2-Base
  - KimiVL-A3B-Instruct

I ran a single image for inference on all 3 models to obtain the result. 

