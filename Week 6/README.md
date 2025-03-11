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

In this case we used [Llava-one-vision Model](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf).

We implemented the model as such
```
# For generating the questions
conversation = [
    {

      "role": "user",
      "content": [
          # Input Prompt 
          {"type": "text", "text": f"Generate different {questions} questions for the image without any answers "},
          {"type": "image"}, 
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Input Image
image_file = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
raw_image
```
where {questions} is the number of questions we want
```
1. What is the dog wearing?
2. What is the woman wearing?
3. What is the setting of the image?
4. What is the time of day?
5. What is the weather like?
6. What is the color of the sand?
7. What is the color of the dog's fur?
8. What is the color of the woman's shirt?
9. What is the color of the dog's collar?
10. What is the color of the woman's hair?
11. What is the color of the waves in the ocean?
12. What is the color of the sky?
13. What is the color of the sand?
14. What is the color of the dog's eyes?
15. What is the color of the woman's shoes?
```
This is what it returns, which has a clear improvement than than Qwen-2b-Instruct model 

However, when I tried to generate more questions, some of the questions were quite repetetive even though I specified that they must be different\
and thus through trial and error that 15 was around the optimal number to generate questions without much repetition 

We then passed this input back to the Qwen-2b-Instruct model which have proven to be quite good at answering specific questions based from the example above
```

1. The dog is wearing a harness.
2. The woman is wearing a plaid shirt.
3. The setting of the image is a beach.
4. The time of day is either sunrise or sunset.
5. The weather is sunny.
6. The color of the sand is beige.
7. The color of the dog's fur is light brown.
8. The color of the woman's shirt is white.
9. The color of the dog's collar is black.
10. The color of the woman's hair is black.
11. The color of the waves in the ocean is white.
12. The color of the sky is light blue.
13. The color of the sand is beige.
14. The color of the dog's eyes is brown.
15. The color of the woman's shoes is black.
```
As we can see from the answers, most of the answers are correct, however some questions are still wrong\
Qn 6 , The color of the woman's shirt is white, when clearly the shirt from the image shows purplish red.\
Qn 14, Where the colour of the eye's dog is clearly not visible ( Hallucination )
Qn 15, The women does not have shoes on ( Hallucination )

We then save this output in a .json file labelled questions_and_answers.json
