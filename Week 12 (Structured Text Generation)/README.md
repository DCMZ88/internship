# Research & Literature 

## Text Generation from Vision Language Models 

- [Problem](#problem)
- [Approaches](#Approaches-taken)
    - [Prompt Engineering](#approach-1-prompt-engineering)
    - [Controlled Text Generation](#approach-2-controlled-text-generation)
    - [Structured Text Generation](#approach-3-structured-text-generation)
- [Validation](#validation)
- [Example](#test-run)
  - [Result](#results)
  - [Conclusion](#conclusion)
- [Example-2](#second-example)
  - [Result](#results) 
### Problem

One challenge that I often came across with the usage of VLMs was getting an consistent output for each inference. In order to conduct evaluation of the performance of VLMs
on its object detection capabilities, I often required the outputs of the VLM to be standardised and consistent in order to compare it to the 'Ground Truth'.

For example, to calculate the recall and precision values of the VLMs ability to detect and list all the objects in an image, a consistent output of the labelled objects was required
in order for me to run the inference over a whole dataset of images so as to parse the output for each inference. 

The aim of the solution for this problem was to ensure that the VLMs generate a consistent JSON format for post-processing tasks such as evaluation.

Example:

Required output : `[dog, woman ,bottle]`

This allows me to write a code to save the strings in the list to a Json file.\
However, this might not be the case.

Unexpected output: `[ Sure! Here are the objects detected in an image: 1. Dog wagging its tail , 2. A woman wearing a Red Dress, 3. Water Bottle ]`

As seen above, it is hard to save the objects labelled in a Json file by using a code due to multitude of factors causing inconsistent outputs\
(Headers, Numbering, Complexity)

Although large language models excel at producing coherent responses, ensuring their outputs respect a specific format is not guaranteed.
Consequently, this can present challenges when utilizing the outputs of a language model as input for another system.
Here we may want to get more structured information than just text back. To use them as software components we need a reliable interface to be able to connect with external tools.
These integrations are possible with friendly serialization formats like JSON or XML.

### Approaches 

#### Approach 1: Prompt Engineering 

One way to counteract is this is by instructing the VLM in its prompt to output in a certain format. 
```
Ensure that the output is in this format:
{ "object": dog , "object": woman }
```
This allows it to automatically generate the output instructions and this have been the primary method I have been using to parse and evaluate the performance of the VLM.

**Limitations**

- Does not guarantee the desired output for every inference as mentioned above.
- Less capable models (i.e Models with smaller parameters) might have difficulty following the instructions. Requires a loop that tries again until the model generates a suitable candidate that is accepted by the parser which requires alot of computational resources and time over a large dataset.
- Having it follow a complex output format may hinder the VLMs capabilities and increase the chance of the VLM making a mistake.

Hence, as the models I have been running have generally smaller parameters and mostly have trouble generating the exact format in the input for parsing. 

#### Approach 2: Controlled Text Generation 

For libraries such as [Jsonformer](https://github.com/1rgs/jsonformer), they are able to perform controlled text generation for LLM where the output will be the format of the input schema.\
In structured data, many tokens are fixed and predictable. Jsonformer is a wrapper around Hugging Face models that fills in the fixed tokens during the generation process, and only delegates the generation of content tokens to the language model. This makes it more efficient and bulletproof than existing approaches.\
(Adapted from Jsonformer GitHub's Repository)

**Limitations**

- Currently libraries such as Jsonformer that peform controlled text generation only supports LLM and not VLMs, which means that this approach is unusable for my case.
- Only works with HuggingFace Models. ( For Jsonformer )
- Only ensures structured output, does not ensure that the model will not hallucinate the values. 

#### Approach 3: Strutured Text Generation

Unlike libraries such as Jsonformer that conforms the output to only be in JSON, libraries such as Outlines enables structured text generation from language models by guiding outputs to match predefined schemas (e.g., using Pydantic). It works by prompting the model, validating the response, and automatically retrying if the output doesn’t conform. This ensures reliable, structured outputs—such as JSON or typed objects—even when working with probabilistic models like GPT-4 or vision-language mode.

Essentially, it allows the model to retry until it validates that the output is in the format of the input schema, which is unlike Jsonformer which forces the output structure to be a JSON format. 

**Limitations**

- Increased computational resources and time required if multiple retries occur
- Schema Complexity: Deep or nested schemas may be harder for the model to consistently generate.
- Limited to Hosted Models ( Outlines )

## Validation 

To validate the outputs from the VLM to ensure that it is consistent in its format, I introduce the use of the library Pydantic.

Pydantic is a data validation and parsing library for Python. It is widely used for ensuring that the data conforms to a specific structure and type. Pydantic is particularly useful in applications where data integrity and correctness are important, such as web APIs, configuration management, or data processing pipelines.

In this case, we can validate whether the output is in the JSON format that we require.
```
from pydantic import BaseModel

class ObjectBox(BaseModel):
    object: str
    bbox: list[float]  # [x1, y1, x2, y2]

class DetectionResult(BaseModel):
    objects: list[ObjectBox]
    counts: dict[str, int]

# Example input ( Output from the VLM )
input_data = {
    "objects": [
        {"object": "dog", "bbox": [0.1, 0.2, 0.3, 0.4]},
        {"object": "cat", "bbox": [0.5, 0.6, 0.7, 0.8]},
    ],
    "counts": {
        "dog": 1,
        "cat": 1
    }
}

result = DetectionResult(**input_data)
print(result.objects[0].object)  # prints "dog"
```
For example, we can validate whether the output of the VLM return the objects and bounding boxes as 2 different key-value pairs in a single dictionary.\
If the output is not in that format, Pydantic would raise an error. 

### Test Run

To test out whether structured text generation can be integrated into VLMs, I decided to use [Outlines](https://github.com/dottxt-ai/outlines) by dottxt.ai, library that enables structured text generation in LLMs and VLMs.

For this case, we will prompt the VLM to detect the objects in a image, and subsequently, output the label of the objects detected, their corresponding bounding boxes and the object count in a specified JSON schema. 

Model used: Qwen2.5-VL-7B-Instruct 

[Code](https://github.com/DCMZ88/internship/blob/main/Week%2012%20(Structured%20Text%20Generation)/Codes/TextGeneration.ipynb)

Prompt used =
```
You are very skilled at detecting simple objects in an image.
Detect all the common, simple objects (e.g., dog, cat, car, chair) and their corresponding bounding boxes coordinates in the image. 
Do not detect complex descriptions (e.g., 'dog in fire', 'cat on a table') or any text present in the image.
Count how many of each objects there are, and return the results in the following JSON schema:
{ImageDetectionResult.model_json_schema()}
```
JSON Schema (Using Pydantic):
```
class DetectedObject(BaseModel):
    label: str = Field(..., description="Label of the detected object")
    bbox: List[int] = Field(..., description="Bounding box in [x1, y1, x2, y2] format")

class ImageDetectionResult(BaseModel):
    objects: List[DetectedObject] = Field(..., description="List of detected objects with labels and bounding boxes")
    object_counts: Dict[str, int] = Field(..., description="Dictionary with the count of each detected object type")
```

With this prompt, I ran the model over a single image to validate whether Outlines could work for VLMs.

<p align="middle">
  <img src="https://github.com/user-attachments/assets/97f336cc-fd8b-44c8-8138-967cfb32b34e", width="500">
  <br>Figure 1: Reference Image
</p>

Expected JSON output
```
{
    "objects": [
        {
            "label": "<label_of_detected_object>",
            "bbox": [<x1>, <y1>, <x2>, <y2>]
        },
        ...
    ],
    "object_counts": {
        "<label_of_detected_object>": <count_of_object>,
        ...
    }
}
```
#### Results

Labelled under 'Structured Text Generation'

Actual Ouput:
```
{
    "objects": [
        {
            "label": "chair",
            "bbox": [160, 139, 331, 378]
        },
        {
            "label": "iron",
            "bbox": [497, 150, 645, 392]
        }
    ],
    "object_counts": {
        "chair": 1,
        "iron": 1
    }
}
```
I then plotted the bounding boxes and labels onto the image to visualise the results by the VLM.

<p align="middle">
  <img src="https://github.com/user-attachments/assets/510a2bec-879f-4bbd-b326-f242a7246278", width="500">
  <br>Figure 2: Processed Image (Qwen2.5)
</p>

#### Conclusion
From the processed image, we can see that the model correctly labels the object identified and outputs the correct bounding boxes of each object. This suggests that the precision is high for this model. However, its accuracy can be quite low as seen from the image. Missing out on a few objects despite telling it to detect all the objects in the image.

However, this also proves that structured text generation works for Qwen2.5 as it outputs in JSON formatting.

### Second Example

However, when tested on a different model, the results were quite different. Using the same prompt and input image to the VLM, these were the results generated.

Model: Gemma3-4b-it

#### Results

Output:
```
{
  "objects": [
    {
      "label": "chair",
      "bbox": [352, 384, 534, 632]
    },
    {
      "label": "table",
      "bbox": [68, 268, 366, 512]
    },
    {
      "label": "iron",
      "bbox": [770, 342, 988, 588]
    },
    {
      "label": "clothing",
      "bbox": [166, 136, 834, 794]
    },
    {
      "label": "pillow",
      "bbox": [143, 770, 312, 976]
    },
    {
      "label": "figurine",
      "bbox": [720, 136, 914, 348]
    }
  ],
  "object_counts": {
    "chair": 1,
    "table": 1,
    "iron": 1,
    "clothing": 1,
    "pillow": 1,
    "figurine": 1
  }
}
```

<p align="middle">
  <img src="https://github.com/user-attachments/assets/6b14d53c-1d8f-4b47-a0e1-304d078a22de", width="500">
  <br>Figure 3: Processed Image (Gemma3)
</p>

As seen from the results, Gemma3 was able to detect more objects than the Qwen2.5, but the bounding boxes were clearly incorrect.
> I also had to resize the image and coordinates as the original size of the image were 436x416 respectively but the coordinates given were normalized to 1024x1024.

I also tried including the prompt to align the coordinates with the input's image resolution but it still gave me pixel coordinates outside of the image's pixel size.

