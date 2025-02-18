# Progress for Week 2
## Data Cleaning & Preprocessing
### Learned how to process videos
Split videos frame by frame through use of OpenCV then loaded it through the OD-model to classify videos
### Identify and clean out duplicated images 
Used phash function from imagehash to compare images's hash value to determine and seive out duplicates 
## Research & Literature Review
### Basics of CNN
**Basic Structure of CNN** : Input -> Convolution , Activation, Pooling -> Fully Connected Layer -> Output
**Convolutional Neural Networks** : Used primarily for object identification, image classification etc.
# Challenges faced
### Data Cleaning & Preprocessing
- Kernel keeps dying whenever the model tries to process over a total of 1100 images 
  even though i tried to process it in batches of 50 even with the use of gpu
- Unable to process long videos and gifs
- Output Video has no sound 
### Research & Literature Review
- Unfamiliar with GitHub and all the processess before being able to code
- Spent alot of time figuring out how to use different platforms and libraries
  i.e setting up virtual environment  
