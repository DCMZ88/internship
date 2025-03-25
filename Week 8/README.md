# Data Curation 
- [Evaluation over a data set](#Evaluation-over-a-data-set)
- [Challenges](#challenges) 
## Evaluation of model's performance

### Methodology 

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
Qwen2.5-VL-Instruct : 52.3%
Llava-hf : 67.0%

## Evaluation over a data set


**Aim** : Calculate each model's performance on a given dataset in its object detection capabilities. 

### Methodology

In order to achieve this, we will just iterate the dataset of images over each VLM to produce labels and pass it through the Object Detector Model.

1. Prompt and Pass each image into the model to produce labels for the corresponding image
2. Save the labels and the corresponding image path to a `json` file.
3. Load the `json` file and pass it through the Owlv2 model ( Object Detection Model )
4. Save the incorrect and correct labels into a `json` file
5. Record the average success rate for each model.

**Results**

256 Images were used for this results. (Images1)

Prompt used 
```
    Detect all the generic objects in the image and list them out,
    Only Include the name of the objects.

    Do not include the header.
    The output should be in this format: "dog , woman, ball"
```


Google's Gemma-3-4b-it : 56.1% 

### Challenges 
- I had initially tried to run both models simultaneously but the kernel crashed while processing the image through the Owlv2 thus I had to run both models separately, creating the need to save the labels from the VLM in a `json` file.
- Another challenge faced was getting the VLMs to output a consistent format\
  For example, Google's Gemma3 Model would sometimes output the labels with the standard heading by the model or the objects would be listed in list with a hyphen which made it hard to use a code to save the labels in a list.
### Limitations 
- Assume that Owlv2 is the "Ground Truth"
- Some of the objects labelled by the VLMs may be too specific though correct which Owlv2 may not detect.
