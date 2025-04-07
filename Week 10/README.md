# Data Analysis

## Evaluation of the performance on VLMs ( cont. )

**Aim** : Evaluate other VLM's performance on object detection and obtain their precision score

**Methodology**

We use the same method in determining the precision scores as that in [Week 9](https://github.com/DCMZ88/internship/tree/main/Week%209), changing only the models used.

**Information**

Visual Language Models used: Qwen2.5-vl-Instruct , Llava-hf-v1.6-mistral-7b\
Zero-shot object detectors: GroundingDINO, OmDet, Owlv2\
Threshold ( For object detection ) : 0.25 

Results 

Overall Precision :
> Using all 3 object detection models
- Qwen2.5-vl-instruct : 25.6
- Llava-hf-v1.6-mistral-7b: 33.2
- Google's Gemma-3-4b-it : 43.3
