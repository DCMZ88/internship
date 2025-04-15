# Data Analysis

## Evaluation of the performance on VLMs (cont.)

### OCR Capabilities 

**Aim** : Evaluate model's performance based on its ability to detect text in an image. 

**Methodology** 

1. Prompt each model to detect text in each image in the dataset
2. Split the outputs into individual words and saves the output in a `json` file for each model
3. Generate 'Ground Truth' using OCR model.
4. Compare Ground Truths with model outputs
6. Calculate relevant metrics

**Second Method to generate Ground Truth**

Generating the Ground Truth:
   1. Comparing the outputs of the VLMs to generate a 'Ground Truth'
   2. If the word/label appears in more than 75% of the models, record it as a 'Ground Truth'

*Code*
1. Generate Outputs from VLMs ( Individual Models labelled 'Upscaled(Model).ipynb' )
2. [Generate Ground Truth from OCR Model](https://github.com/DCMZ88/internship/blob/main/Week%2011/Codes/PaddleOCR.ipynb)
3. [Compare and Evaluate Outputs with Ground Truth](https://github.com/DCMZ88/internship/blob/main/Week%2011/Codes/FinalEvaluation.ipynb)

[Generate Ground Truth by comparing outputs of VLMs](https://github.com/DCMZ88/internship/blob/main/Week%2011/Codes/Consensus_Ground_Truth.ipynb)

**Metrics used to evaluate OCR capabilities**

> Taken from [OCR Accuracy](https://www.docsumo.com/blogs/ocr/accuracy)

1. Character Error Rate ( CER )

2. Precision & Recall Values 
   True Positives = Correct Words\
   False Positives = Wrong Words\
   False Negatives = Missed out Words

<p align="middle">
  <img src="https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/639c3cc56bda8713d4a2f29c_precision-recall.webp" width=250, height=200>
  <br>Figure 1: Precision-Recall 


### Challenges
   - **Determining the False Positives** : I split all the text into individual words so it is a list of words for one image which poses a challenge to compare the individual words to its 'Ground Truth'.\
   i.e `Ground Truth: [dog, apple , rvd , green ]` and `Model Output: [ apple, red, green,]`\
As seen from the example it is hard to map 'rvd' to 'red' to label it as a False Positive.

   - **Handling Repeated Words** : If the word appears twice in the output of the VLM, how to ensure that one of it is labelled as the False Positive if the it only appears in the 'Ground Truth' once.\
     E.g `Ground Truth: Dog`, `Output: Dog, Dog`
     
### Limitations 

   - **Consistency of VLM output** : The outputs of the VLMs tend to vary from image to image.\
    E.g Some outputs of for certain images contains a header from the VLM and some does not.

   - **Special Symbols** : Special symbols detected by the OCR model causes the words detected to vary.\
    E.g `Ground Truth : [ "17Jul20238:01pm" ]` ,`VLM Output : [ "17", "Jul" ,"2023" ,"8:01pm"]`\
    Therefore, it is hard to separate the words consistently using code across the whole dataset, leading to inaccurate evaluation of the model.
