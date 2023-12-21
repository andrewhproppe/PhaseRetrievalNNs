from PRNN.utils import get_system_and_backend
get_system_and_backend()

import cv2
import os
import shutil
from tqdm import tqdm

def is_non_green_flower(image_path, green_threshold=0.2):
    image = cv2.imread(image_path)

    if image is not None:
        # Convert BGR to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Threshold the image to identify green regions
        green_mask = cv2.inRange(hsv_image, (40, 40, 40), (80, 255, 255))
        green_pixel_count = cv2.countNonZero(green_mask)

        # Calculate the ratio of green pixels to total pixels
        total_pixel_count = hsv_image.size
        green_ratio = green_pixel_count / total_pixel_count

        # Check if the image has a low ratio of green pixels
        return green_ratio < green_threshold

    else:
        print(f"Error: Unable to read the image at {image_path}")
        return False

def filter_green_flower_images(directory, output_directory, green_threshold=0.2):
    image_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])

    files_moved = 0

    for image_file in tqdm(image_files):
        image_path = os.path.join(directory, image_file)

        if not is_non_green_flower(image_path, green_threshold):
            # Move the image to the output directory
            output_path = os.path.join(output_directory, image_file)
            shutil.move(image_path, output_path)
            files_moved += 1

    return files_moved


# Example usage
input_directory = "masks/flowers_testing"  # Replace with the path to your image directory
output_directory = "masks/flowers_testing_greens"   # Replace with the desired output directory

# Set a threshold for the green ratio (experiment with values)
green_threshold = 0.25

files_moved = filter_green_flower_images(input_directory, output_directory, green_threshold)

print(f"{files_moved} files moved to {output_directory}")

# import cv2
# import os
# import numpy as np
#
# from matplotlib import pyplot as plt
#
# def calculate_pixel_sum(image_path):
#     image = cv2.imread(image_path)
#
#     if image is not None:
#         # Split the image into its RGB channels
#         b, g, r = cv2.split(image)
#
#         # Calculate the sum of red, green, and blue pixels
#         red_sum = np.sum(r)
#         green_sum = np.sum(g)
#         blue_sum = np.sum(b)
#
#         return red_sum, green_sum, blue_sum
#
#     else:
#         print(f"Error: Unable to read the image at {image_path}")
#         return 0, 0, 0
#
# def process_images_in_directory(directory, fraction_threshold):
#     red_sums = []
#     green_sums = []
#     blue_sums = []
#     selected_indices = []
#
#     image_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
#
#     for index, image_file in enumerate(image_files):
#         image_path = os.path.join(directory, image_file)
#         red_sum, green_sum, blue_sum = calculate_pixel_sum(image_path)
#
#         red_sums.append(red_sum)
#         green_sums.append(green_sum)
#         blue_sums.append(blue_sum)
#
#         # Check if green sum is higher than both red and blue sums by a fraction
#         if green_sum > fraction_threshold * red_sum and green_sum > fraction_threshold * blue_sum:
#             selected_indices.append(index)
#
#     return red_sums, green_sums, blue_sums, selected_indices
#
# # Example usage
# directory_path = "masks/flowers_testing"  # Replace with the path to your image directory
# fraction_threshold = float(input("Enter the fraction threshold (e.g., 1.2): "))
#
# red_sums, green_sums, blue_sums, selected_indices = process_images_in_directory(directory_path, fraction_threshold)
#
# print("Sum of Red Pixels:", sum(red_sums))
# print("Sum of Green Pixels:", sum(green_sums))
# print("Sum of Blue Pixels:", sum(blue_sums))
# print(f"Indices where Green sum is higher than both Red and Blue sums by {fraction_threshold}: {selected_indices}")
#
# import time
#
# def display_selected_images(directory, selected_indices):
#     image_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
#
#     # Calculate the number of rows and columns based on the length of selected_indices
#     num_images = len(selected_indices)
#     ncols = 3  # Set the number of columns
#     nrows = (num_images + ncols - 1) // ncols  # Calculate the number of rows
#
#     # Create subplots
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
#
#     for i, index in enumerate(selected_indices):
#         if 0 <= index < len(image_files):
#             # Calculate subplot indices
#             row = i // ncols
#             col = i % ncols
#
#             # input('Press any button to go to next image')
#             image_file = image_files[index]
#             image_path = os.path.join(directory, image_file)
#
#             # Display the image corresponding to the selected index using Matplotlib
#             selected_image = cv2.imread(image_path)
#             selected_image = cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#
#             # Plot the image in the current subplot
#             axes[row, col].imshow(selected_image)
#             axes[row, col].set_title(f"Selected Image {index}")
#             axes[row, col].axis('off')
#
#     # Adjust layout and display the subplots
#     plt.tight_layout()
#     plt.show()
#
# display_selected_images(directory_path, selected_indices)
#
