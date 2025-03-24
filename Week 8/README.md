# Data Curation 

## Evaluation of model's performance

## Methodology 

I decided to try to use another zero-shot object detector model rather than GroundingDINO as the accuracy of GroundingDINO was good enough for it\
to act as the "Ground Truth"

For this attempt, I used Google's [owlv2-base-patch16-ensemble](#https://huggingface.co/google/owlv2-base-patch16-ensemble)

Utilising the same methodology in Week 7,\
Prompt the VLMs for labels through object detection\
Pass these labels through Owlv2\
See if Owlv2 is able to identify the labels from the image.

We set the threshold for the confidence score to > 0.5.

**Results**

Google's Gemma-3-3B-it : 55.8% 
