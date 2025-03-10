# Data Curation 

## Methodology

 **Aim** : Generate different inference questions for each image through the use of a VLM.

For this project, I used the [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) model adapted from huggingface. 

Model Input : Image , Text

Model Output : Text 

This allows us to input an image with a text prompt to generate an output text as follows 

```
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": """Generate 20 simple questions about the image for image inferencing
             or object detection that is clearly visible in the image for other visual language models to infer,
             give me only the questions"""},
        ],
    }
]
```
