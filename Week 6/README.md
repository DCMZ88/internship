# Data Curation 

## Methodology

 **Aim** : Generate different inference questions for each image through the use of a VLM.

For this project, I used the [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) model adapted from huggingface. 

Note:
I tried using DeepSeek-vl2 model but the parameters were too large for the GPU to handle and thus kept forcing the kernel to crash unexpectedly as it required at least 80GB of GPU memory

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
In this case, we want to generate 20 questions based on the image as seen in the "text" prompt. 

<p align="center">
  <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" width="500" />
  <br>Figure 1: Input Image
</p>

We then obtain this output from the model 
```
["1. What is the woman doing?
\n2. What is the dog doing?
\n3. What is the dog wearing?
\n4. What is the woman wearing?
\n5. What is the weather like?
\n6. What is the time of day?
\n7. What is the dog's breed?
\n8. What is the woman's hairstyle?
\n9. What is the woman's shoe?
\n10. What is the dog's leash?
\n11. What is the dog's collar?
\n12. What is the dog's harness?
\n13. What is the dog's tail?
\n14. What is the dog's front paw?
\n15. What is the dog's back paw?
\n16. What is the dog's body posture?
\n17. What is the dog's facial expression? \n18. What is the dog's tail color?
\n19. What is the dog's fur color? \n20. What is the dog's breed?"]
```
As we can see the questions generated are somewhat repetitive ( n7 and n20 : What is the dog's breed )\
and that some questions are hard to answer even for a human.\
(e.g. n13\What is the dog's tail?)

Next, I tried to feed the output questions back into the model.\
This is what i got.
```
["1. The woman is sitting on the beach, smiling and playing with the dog.
\n2. The dog is sitting on the beach, with its front paw raised in a high-five gesture.
\n3. The dog is wearing a harness.
\n4. The woman is wearing a plaid shirt and black pants.
\n5. The weather appears to be sunny and warm, with a clear sky.
\n6. It is likely early morning or late afternoon, as the sun is low in the sky.
\n7. The dog's breed is not clearly identifiable from the image.
\n8. The woman's hairstyle is not visible in the image.
\n9. The woman is not wearing shoes.\n10. The dog's leash is red.
\n11. The dog's collar is not visible in the image.
\n12. The dog's harness is blue and red.
\n13. The dog's tail is long and bushy.
\n14. The dog's front paw is raised in the air.
\n15. The dog's back paw is not visible in the image.
\n16. The dog's body posture is relaxed and playful.
\n17. The dog's facial expression is happy and playful.
\n18. The dog's tail is a light brown color.
\n19. The dog's fur color is light brown.
\n20. The dog's breed is not clearly identifiable from the image."]
```
As we can see from the output text, the VLM struggles to answer some of its own questions if the questions are too specific and is not able to identify it.\
However, for the most part the model seems to be able to correctly interpret and answer its own questions.

Next, even though I changed the prompt of the question a few times, the questions being generated were still too specific despite prompting the VLM.\
Hence I used a different VLM to generate the questions.

In this case we used [Microsoft's Florence-2-base](https://huggingface.co/microsoft/Florence-2-base).
