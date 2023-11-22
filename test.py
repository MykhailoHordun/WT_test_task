from model import *
from utils import *
import numpy as np
from PIL import Image  
import os
import cv2
import matplotlib.pyplot as plt

weights = './unet.h5' # Replace with the path for the pretrained U-net model
model = unet_model(pretrained_weights=weights) 
input_shape = model.input_shape[1:]  # Exclude batch size

# Iterate over the images in the folder
image_folder = './test'  # Replace with the path to your image folder
mask_folder = './masks' # Replace with the path to your mask folder or set Fasle if don`t have target masks
image_files = os.listdir(image_folder)

output_dir = './pred_masks'  # Replace with the path for the output directory
os.makedirs(output_dir, exist_ok=True)

pred = []
for file_name in image_files:
    # Load and preprocess the image
    image_path = os.path.join(image_folder, file_name)
    image = Image.open(image_path)
    image = image.resize(input_shape[:2])  # Resize the image to match the input shape
    image = np.array(image) / 255.0  # Normalize pixel values (optional, based on model requirements)
    image = np.expand_dims(image, axis=0)  # Add a batch dimension

    if mask_folder:
        mask_path = os.path.join(mask_folder, file_name)
        mask = Image.open(mask_path)
        mask = mask.resize(input_shape[:2])  # Resize the mask to match the input shape
        mask = np.array(mask) / 255.0  # Normalize pixel values (optional, based on model requirements)
        mask = np.expand_dims(mask, axis=0)  # Add a batch dimension

        print(file_name)
        model.evaluate(image, mask)
    else:
        print(file_name)
        predicted_mask = model.predict(image)
        # Convert the predicted mask to the appropriate data type
        predicted_mask = np.squeeze(predicted_mask, axis=0)  # Remove the batch dimension
        # Generate the output file path
        output_path = os.path.join(output_dir, file_name)
        # Save the predicted mask
        cv2.imwrite(output_path, predicted_mask * 255.0)
        pred.append(predicted_mask)
# plotting of image and mask

if not mask_folder:
    n = 5 # number of image from set you want to see

    file_name = image_files[n]
    image_path = os.path.join(image_folder, file_name)
    img = Image.open(image_path)

    fig, ax = plt.subplots(1, 2, figsize=(8,8))
    ax[0].imshow(img)
    ax[0].set_title('Real image')
    ax[1].imshow(pred[n])
    ax[1].set_title('Predicted Mask')

    plt.show()
