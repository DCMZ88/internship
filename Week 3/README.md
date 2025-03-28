# Progress for Week 3
## Data Cleaning and Preprocessing
### Sort Videos
Configured the ObjectDetector(Videos) script to be able to automatically detect humans and sort them into respective folders (Non-Human) and (Human). (Code is in [SortVideos.ipynb](https://github.com/DCMZ88/internship/edit/main/Week%203/SortVideos(Updated).ipynb)).

How it works:
- Goes through each file in the folder
- In each video file, processes the video into frames and stores it in an output_frames folder
- Passes these images(frames) into the model to output the detections
- Loops through each detection in each image to check if there is human inside each frame
- Once a human is detected in the frame, the video is automatically sorted to human_vid , Else, after looping through all the frames and no human was detected, automatically sort it to the nonhuman_vid folder.
- Repeats for each video in the folder until all videos have been sorted.
## Research and Literature Review 
### Basics of ML Algorithms 
- Logistic Regression
- Linear Regression
- Neural Networks
### Architecture of ML Algorithms
- Loss Function
- Gradient Descent
- Regularization
