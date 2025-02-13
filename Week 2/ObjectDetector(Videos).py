#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import pipeline ,YolosImageProcessor, YolosForObjectDetection
from PIL import Image,ImageDraw, ImageFont
import requests
from IPython.display import display
import cv2 
import numpy as np
import sys
import os
import torch
import shutil

# Load Object-Detection Model
model = pipeline(model="hustvl/yolos-tiny", device="cuda:0")


# In[ ]:


# For deleting folders
def delete_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' has been deleted successfully.")
    else:
        print(f"Error: Folder '{folder_path}' does not exist.")

# Example usage
delete_folder('output_frames')
delete_folder('human')
delete_folder('nothuman')


# In[ ]:


# Video capture
cap = cv2.VideoCapture("test.mp4")

# Output folder for saving frames
output_folder = "output_frames"
os.makedirs(output_folder, exist_ok=True)
# Labels
frame_count = 0

while True:
    # Reads each frame 
    success, img = cap.read()
    if not success:
        break  # Break out of the loop if there are no more frames to read

    # Save the current frame as an image to output folder
    output_path = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
    cv2.imwrite(output_path, img)
    
    frame_count += 1

# Release video capture
cap.release()
# Number of frames in the video
print(f"Total frames saved: {frame_count}")


# In[ ]:


# Loads the images
def load_images_from_folder(folder_path):
    images = []
    file_name = []
    # Goes through the folder
    for filename in os.listdir(folder_path):
        # If the file is an image
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Add any other image file extensions you need
            image_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(image_path)
                # Adds the image to the images list 
                images.append(image)
                # Records the file-name loaded
                file_name.append(filename)
            except Exception as e:
                print(f"Failed to load image {filename}: {e}")
    return images, file_name

# Split list into chunks to ensure processor doesnt crash
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

# Ensure output directories exist
if not os.path.exists('human'):
    # Makes directories if it doesn't exist 
    os.makedirs('human')
if not os.path.exists('nothuman'):
    os.makedirs('nothuman')
# Folder path of the images
folder_path = '/home/jovyan/output_frames'
# Loads the images to a list 
loaded_images,file_name = load_images_from_folder(folder_path)


# In[ ]:


# Process images in batches
chunk_size = 50
index = 0
# Loops for every 50 images
for i, image_chunk in enumerate(chunk_list(loaded_images, chunk_size)):
    print(f"Processing batch {i + 1}")
    # Sends 50 images to the AI model to process
    detections = model(image_chunk)
    # For each image
    for j in range(len(image_chunk)):       
        success = False
        while index < len(file_name):
            # Determines the original image path
            filename = file_name[index]
            image_path = os.path.join(folder_path, filename)
            if os.path.isdir(image_path):  # Skip directories
                index += 1
                continue
            image = Image.open(image_path)
            img_array = np.asarray(image, dtype=np.uint8)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # Loops through labels
            for detection in detections[j]:
                # If a person is detected in the image, record image as success
                if detection['score'] > 0.9 and detection['label'] == "person":
                    success = True
                    x1, y1, w, h = detection['box']['xmin'], detection['box']['ymin'], detection['box']['xmax'], detection['box']['ymax']
                    label = "Human"
                    score = f"{(detection['score'] *100):.1f}%"
                    text = label + " " + score 

                    # Draw the rectangle around the object
                    cv2.rectangle(image, (x1, y1), (w, h), (0, 0, 0), 1)  # Green rectangle, thickness=2

                    # Add the label text above the box
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, text , (x1, y1 - 10), font, 1.2, (0, 0, 0), 2)  # Green text
                    break
            # Outputs the image to either human or nothuman directory 
            if not success:
                output_path = f'nothuman/{filename}'
            else:
                output_path = f'human/{filename}'
            outcome = cv2.imwrite(output_path, image)
            index += 1
            break
print("Processing complete.")


# In[ ]:


# Recreates video 


# In[ ]:


output_folder = '/home/jovyan/human'


# In[ ]:


# Recreate video from saved frames
# Output path
output_video_path = "output_video.mp4"
img_array = []
# Goes through each frame in the folder 
for filename in sorted(os.listdir(output_folder)):
    if filename.endswith(".jpg"):
        img_path = os.path.join(output_folder, filename)
        img = cv2.imread(img_path)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
# Arranges the frame to a video 
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

print("Video created successfully.")


# In[ ]:


# With boxes and labels


# In[ ]:


# Process images in batches
chunk_size = 50
index = 0
# Loops for every 50 images
for i, image_chunk in enumerate(chunk_list(loaded_images, chunk_size)):
    print(f"Processing batch {i + 1}")
    # Sends 50 images to the AI model to process
    detections = model(image_chunk)
    # For each image
    for j in range(len(image_chunk)):       
        success = False
        while index < len(file_name):
            # Determines the original image path
            filename = file_name[index]
            image_path = os.path.join(folder_path, filename)
            if os.path.isdir(image_path):  # Skip directories
                index += 1
                continue
            image = Image.open(image_path)
            img_array = np.asarray(image, dtype=np.uint8)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # Loops through labels
            for detection in detections[j]:
                # If a person is detected in the image, record image as success
                if detection['score'] > 0.9 and detection['label'] == "person":
                    success = True
                    x1, y1, w, h = detection['box']['xmin'], detection['box']['ymin'], detection['box']['xmax'], detection['box']['ymax']
                    label = "Human"
                    score = f"{(detection['score'] *100):.1f}%"
                    text = label + " " + score 

                    # Draw the rectangle around the object
                    cv2.rectangle(image, (x1, y1), (w, h), (0, 255, 0), 1)  # Green rectangle, thickness=2

                    # Add the label text above the box
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, text , (x1, y1 - 10), font, 1.0, (0, 255, 0), 2)  # Green text
                    break
            # Outputs the image to either human or nothuman directory 
            if not success:
                output_path = f'nothuman/{filename}'
            else:
                output_path = f'human/{filename}'
            outcome = cv2.imwrite(output_path, image)
            index += 1
            break
print("Processing complete.")

