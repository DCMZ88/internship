# Data Analysis 
- [Evaluation over a data set](#Evaluation-over-a-data-set)
    - [Challenges](#challenges)
    - [Limitations](#limitations)
## Evaluation of model's performance

### Methodology 

I decided to try to use another zero-shot object detector model rather than GroundingDINO as the accuracy of GroundingDINO was good enough for it
to act as the "Ground Truth"

For this attempt, I used Google's [owlv2-base-patch16-ensemble](#https://huggingface.co/google/owlv2-base-patch16-ensemble)

Utilising the same methodology in Week 7,\
Prompt the VLMs for labels through object detection\
Pass these labels through Owlv2\
See if Owlv2 is able to identify the labels from the image.

We set the threshold for the confidence score to > 0.5.

**Results**

Google's Gemma-3-3B-it : 55.8%\
Qwen2.5-VL-Instruct : 52.3%\
Llava-hf : 67.0%

## Evaluation over a data set


**Aim** : Calculate each model's performance on a given dataset in its object detection capabilities. 

### Methodology

In order to achieve this, we will just iterate the dataset of images over each VLM to produce labels and pass it through the Object Detector Model.

1. Prompt and Pass each image into the model to produce labels for the corresponding image
2. Save the labels and the corresponding image path to a `json` file.
3. Load the `json` file and pass it through the Owlv2 model ( Object Detection Model )
4. Save the incorrect and correct labels into a `json` file labelled `Labels`
5. Record the average success rate for each model.

**Results**

256 Images were used to attain these results. (Images1)

Prompt used 
```
    Detect all the generic objects in the image and list them out,
    Only Include the name of the objects.

    Do not include the header.
    The output should be in this format: "dog , woman, ball"
```


Google's Gemma-3-4b-it : 56.1% 

Llava's-hf one-vision : 50.5%

Qwen2.5-VL-3B-Instruct : 39.0%

### Challenges 
- I had initially tried to run both models simultaneously but the kernel crashed while processing the image through the Owlv2 thus I had to run both models separately, creating the need to save the labels from the VLM in a `json` file.
- Another challenge faced was getting the VLMs to output a consistent format\
  For example, Google's Gemma3 Model would sometimes output the labels with the standard heading by the model or the objects would be listed in list with a hyphen which made it hard to use a code to save the labels in a list.
- Example
```
- Dog
- Ball
```
- Other models would also struggle to comprehend the task prompted to them, resulting in alot of hallucinations or invalid answers . But when prompted a single task to detect the image,\
it is able to generate a detailed caption of the image quite accurately.
- For the Qwen model, I had to resize the images to a smaller scale as some of the images were too large which caused a memory error.
### Limitations 
- Assume that Owlv2 is the "Ground Truth"
- Some of the objects labelled by the VLMs may be too specific though correct which Owlv2 may not detect.
- Some of the objects detected in some images by the VLMs are vastly incorrect even though I already fine-tuned the prompt which affects the accuracy of the model greatly.
- One thing to note is also that the parameters for each model were different which may affect the overall performance of the model
- The performance of the model may also be due to its ability to understand the prompt well which may not necessarily reflect its Object Detection Capabilities.
