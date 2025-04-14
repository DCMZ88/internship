# Data Analysis

## Evaluation of the performance on VLMs (cont.)

### OCR Capabilities 

**Aim** : Evaluate model's performance based on its ability to detect text in an image. 

**Methodology** 

1. Prompt each model to detect text in each image in the dataset
2. Save the output texts in a `json` file for each model
3. Generate 'Ground Truth' using OCR model.
4. Compare Ground Truths with model outputs
6. Calculate Precision & Recall Values.

**Metrics used to evaluate OCR capabilities**

> Taken from [OCR Accuracy](https://www.docsumo.com/blogs/ocr/accuracy)

1. Character Error Rate ( CER )

2. Precision & Recall Values 
