# Data Analysis

## Evaluation of the performance on VLMs ( cont. )

### Object Detection Capabilities 

**Aim** : Evaluate other VLM's performance on object detection and obtain their precision score

**Methodology**

We use the same method in determining the precision scores as that in [Week 9](https://github.com/DCMZ88/internship/tree/main/Week%209), changing only the models used.

**Information**

Visual Language Models used: Qwen2.5-vl-Instruct , Llava-hf-v1.6-mistral-7b\
Zero-shot object detectors: GroundingDINO, OmDet, Owlv2\
Threshold ( For object detection ) : 0.25 

**Final Results**

Overall Precision :
> Using all 3 object detection models to generate the 'Ground Truth'
- Qwen2.5-vl-instruct : 25.6
- Llava-hf-v1.6-mistral-7b: 33.2
- Google's Gemma-3-4b-it : 43.3
    
Overall Recall :
- Qwen2.5-vl-instruct : 41.4
- Llava-hf-v1.6-mistral-7b: 55.8
- Google's Gemma-3-4b-it : 43.8

### OCR Capabilities

**Aim** : Evaluate model's performance based on its ability to detect text in an image. 

**Methodology** 

1. Prompt each model to detect text in each image in the dataset
2. Save the output texts in a `json` file for each model
3. Generate 'Ground Truth' using OCR model.
4. Compare Ground Truths with model outputs
6. Calculate Precision & Recall Values.

# Research & Literature 

## OCR

OCR stands for Optical Character Recognition, which is a technology that enables the conversion of text in images or scanned documents into editable, machine-readable text. By analyzing visual representations of characters, OCR identifies patterns and translates them into textual information that computers can process.

### Finding OCR Models

OCR Models : PaddleOCR, EasyOCR, Florence-base
> trOCR ( I tried to get it running but the weights could not be loaded from huggingface website leading to very erratic results hence I did not include it here )

I ran all 3 models on an single inference image to detect text in the image to test its capabilities

<p align="middle">
  <img src="https://github.com/user-attachments/assets/27c52ea8-4c5a-417c-9e1d-b5695e0ad270" width=500, height=400>
  <br>Figure 1: Sample Image (Resized for fitting purposes)

**Outputs**

PaddleOCR : 
```
('Joy of', 0.9997420907020569)
('learning', 0.9939911365509033)
('Singaporean', 0.9986035823822021)
('students', 0.9963444471359253)
('Joy of', 0.9978846907615662)
('learning', 0.9936026334762573)
('Parents', 0.9976922273635864)
('turning', 0.9968897700309753)
('everything', 0.9972988367080688)
('Singaporean', 0.9973495006561279)
('into a', 0.9768469929695129)
('students', 0.9961910843849182)
('competition', 0.9973574280738831)
```
EasyOCR:
```
['of', 'learning', 'pngaporean', 'Suudents', 'Joy of', 'learning', 'Parents', 'turning', 'everything', 'Shgaporean', 'into a', 'uudents', 'competition_', 'Joy']
```
Florence-base:
```
{'<OCR>': 'Joy oflearningSingaporeanstudentsJoy ofLearningParentsturningeverythingSingaporeantinto astudentscompetition'}
```

We can see from the results that PaddleOCR and Florence-base had the correct output whereas EasyOCR struggled in determining the correct letters.

However Florence-base is more of a visual foundational model with multiple tasks such as Object Detection, Region Segmentation.\

' Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks.'
- Taken from the [huggingface](https://huggingface.co/microsoft/Florence-2-base)

Hence, to solely serve the purpose for OCR, we will use PaddleOCR which is trained solely for OCR purposes. 

### Metrics to evaluate OCR performance. 
> Taken from [OCR Accuracy](https://www.docsumo.com/blogs/ocr/accuracy)

1. Character Error Rate ( CER )
