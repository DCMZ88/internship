# Data Curation 

## Methodology

 **Aim** : Generate different inference questions and its corresponding answers for each image through the use of a VLM.

To achieve this, I tried implmenting different models together for each the generation of questions and the generation of answers.

## Attempts 
- [First Attempt](#first-attempt)
- [Second Attempt](#second-attempt)
- [Third Attempt](#third-attempt)
- [Third-Attempt(2.0)](#third-attempt-20)
- [Fourth-Attempt](#fourth-attempt)

- [Challenges](#challenges)

## First Attempt ( Qwen2-VL-2B-Instruct )

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

## Second Attempt 
**( Llava-onevision-Qwen2-0 + Qwen2-VL-2B-Instruct )**

In this case for generation of questions, we used [Llava-one-vision Model](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)\
while keeping the Qwen2 model for the answers.

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
This is what it returns, which has a clear improvement than than Qwen-2b-Instruct model, but there was still repetition

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
Qn 14, Where the colour of the eye's dog is clearly not visible ( Hallucination )\
Qn 15, The women does not have shoes on ( Hallucination )

We then save this output in a .json file labelled [questions_and_answers(Qwen).json](https://github.com/DCMZ88/internship/blob/main/Week%206/Q&A/questions_answers(Qwen).json)

## Third Attempt 
**( Llava-onevision-Qwen2-0 + BLIP )**

Now instead of the Qwen model, we try to run it on the [BLIP](https://huggingface.co/Salesforce/blip-vqa-base) by SalesForce. 

The input questions are still the same as before.\
This time, the model outputs words as opposed to sentences as seen in the Qwen model
```
['harness', 'plaid shirt', 'beach', 'sunset', 'sunny', 'white', 'tan', 'red and white', 'blue', 'brown', 'white', 'white', 'white', 'black', 'black']
```
It is also able to identify correctly for most of the questions but still hallucinates\
For Qn 15, where the model answers that the women's shoes is black but infact she does not have any shoes on.

We then save this output in a .json file labelled [questions_and_answers(Qwen).json](https://github.com/DCMZ88/internship/blob/main/Week%206/Q&A/questions_answers(Blip).json)

### Third Attempt 2.0
Now, I modified the prompt through the use of ChatGPT 

However, one thing to note is that even though I did specify no questions are not to be repeated, the VLM struggled to understand\
and I had to tweak the prompt multiple times to evnetually remove all repetitions.

```
text_prompt = "Analyze the image and generate  15 different and simple questions, one at a time, that can be easily answered by looking at the image. Ensure no questions are repeated."
```

This prompt returns
```
1. What is the time of day in the image?
2. What is the color of the dog's fur?
3. Is the dog sitting on the sand or on the beach?
4. What is the color of the woman's shirt?
5. Is the dog wearing a harness?
6. Is the dog sitting or standing?
7. What is the woman doing?
8. Is the dog looking at the woman?
9. Is the woman smiling?
10. What is the color of the sand?
11. Is the dog panting?
12. Is the woman wearing a watch?
13. What is the color of the waves in the background?
14. Is the sky clear or cloudy?
15. What is the weather like in the image?
```
Drawing from this, we can easily see the complexity of the questions generated when using a more structured prompt\
This allows us to generate more questions with depth rather than questions of the same category (i.e colour of x )

Passing this through the BLIP model yields
```
['sunset', 'tan', 'sand', 'red and white', 'yes', 'sitting', 'petting dog', 'yes', 'yes', 'white', 'no', 'yes', 'white', 'clear', 'sunny']
```
By comparing the answers and the questions, all of the questions have been correctly.

We then output the answer-question pair in a .json file , labelled [questions_and_answers(Blip)(2.0)](https://github.com/DCMZ88/internship/blob/main/Week%206/Q&A/questions_answers(Blip)(2.0).json)

So far, this method has proven to be the most accurate, but we have yet to test it out on other images.

## Fourth Attempt 
**Qwen2.0-VL + BLIP**\
Now, instead of only changing the model, I configured the prompt to be more precise yet diverse. I configured the prompt with the help of ChatGPT to generate the prompt below
```
prompt = f"""
Given the input image, generate 15 unique and diverse questions that can be answered based on the visual content.
The questions should cover a wide range of topics such as the following:

The main subject or objects in the image.
Colors, patterns, and shapes.
The actions or movements taking place.
The spatial relationships between objects or people.
Environmental context (indoor, outdoor, nature, urban, etc.).
Emotions, expressions, or moods conveyed.
Any interactions between objects, people, or elements.
Specific details about the setting or background.
Objects or items in the foreground and background.
The condition or state of any objects or people (e.g., new, old, active, idle).
Make sure to vary the types of questions so they touch on different aspects of the image, and ensure that the questions are easily inferable from the visual content.
Ensure each question only has one question to it.
"""
```
This prompt ensures that the VLM is able to cover a wide range of topics such that the questions generated are not similar or repetetive to one another,\
which seemed to the problem when generating more questions such as asking the colour of different objects, e.t.c.

This seemed to yield much more diverse and verbose questions when prompted into the Qwen model.
```
What is the primary subject of the image?
What colors are prominent in the image?
What shapes are present in the image?
What actions are taking place in the image?
What is the spatial relationship between the woman and the dog?
What is the environmental context of the image?
What emotions or expressions are conveyed in the image?
What interactions are taking place between the woman and the dog?
What specific details about the setting or background are visible in the image?
What objects or items are in the foreground and background of the image?
What is the condition or state of any objects or people in the image?
How does the image convey the condition or state of the woman and the dog?
What is the condition or state of the dog in the image?
How does the image convey the condition or state of the woman's mood or expression?
What is the condition or state of the dog's fur or coat in the image?
```

When these questions are passed through the BLIP model, this yields
```
['dog', 'white and brown', 'dog', 'sitting', 'sitting', 'beach', 'happiness', 'friends', 'beach', 'woman and dog', 'people are in beach', 'happy', 'sad', 'happy', 'wet']
```

Though simple answers to complex questions, it is mostly correct.

## Challenges

However, the main challenge is still the repetition when more questions are generated the repetition becomes more evident.

For example, I tried to generate 30 questions using the Qwen2-VL model, this is what it attained.\
Note: this list starts from Question 13
```
What is the condition or state of any objects or people in the image?
What is the condition or state of the dog in the image?
What is the condition or state of the woman in the image?
What is the condition or state of the beach in the image?
What is the condition or state of the ocean in the image?
What is the condition or state of the sky in the image?
What is the condition or state of the woman's hair in the image?
What is the condition or state of the woman's clothing in the image?
What is the condition or state of the dog's collar in the image?
What is the condition or state of the dog's leash in the image?
What is the condition or state of the dog's tail in the image?
What is the condition or state of the dog's paws in the image?
What is the condition or state of the dog's eyes in the image?
What is the condition or state of the dog's ears in the image?
What is the condition or state of the dog's nose in the image?
What is the condition or state of the dog's mouth in the image?
What is the condition or state of the dog's tongue in the image?
What is the condition or state of the dog's fur in the image?
```
Clearly there is a repetition of the condition or state of an object in the image and that it starts to be too specific that even we\
can't tell such as "What is the condition or state of the dog's tongue in the image".

Ideally, through multiple trial and errors, the optimal questions to be generated is around 15 to 18. 


