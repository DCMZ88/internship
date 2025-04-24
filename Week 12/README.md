# Research & Literature 

## Controlled Text Generation 

**Problem** 

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

### Approaches taken

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

