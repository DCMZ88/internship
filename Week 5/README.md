
# Week 5 Progress

## Research & Literature Review 
- Started on the basics of PyTorch models, learning what each line of code does
- Had many problems in setting up VSCode due to the restriction on TechNet
- Solve dependency and package issues for models on huggingface
- Set up the the environment to finally load and run models smoothly on JupyterLabs.


## Challenges Faced:
- As the DSO TechNet has many security restrictions, one being the SSL Certificate error when trying to download extensions of VScode on AIStacks,\
  I had to seek help from mentors in trying to solve such errors.
- Another challenged faced was my unfamiliarity with PyTorch and Huggingface models which proved to be a daunting task when trying to load models and run them on the GPU
- Steps :
    1. Clone the repository using ( git clone https.//repo.git )
    2. Install the required dependencies ( pip install -e .)
    3. Install the requirement.txt ( pip install -r requirements.txt)
    4. Install any additional packages specified by the author
    5. Ensure that the model loads to GPU ( specify model = ("model").to("cuda") )
- Another challenge was that some of the packages were older versions and have degraded which resulted in the functions not working as intended which I had to solve individually
