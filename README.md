# Snaps Mask Fill - ComfyUI Custom Node

A ComfyUI custom node that automatically detects white box areas in a base image and fills them with a second image, intelligently fitting the content to the detected dimensions.

## Features

- **Automatic White Box Detection**: Uses computer vision to find the largest white rectangular area in the base image
- **Smart Image Fitting**: Multiple fitting modes to handle different aspect ratios
- **Configurable White Threshold**: Adjust sensitivity for detecting white areas
- **Preserves Image Quality**: Uses high-quality resampling for image resizing

## Installation

1. Copy this folder to your ComfyUI custom nodes directory:
   ```
   ComfyUI/custom_nodes/Snaps-Mask-Fill/
   ```

2. Restart ComfyUI

3. The node will appear in the "image/processing" category as "Snaps Mask Fill"

## Usage

### Inputs

- **base_image**: The image containing the white box area where content should be placed
- **fill_image**: The image to be placed inside the white box
- **white_threshold**: Sensitivity for detecting white areas (0.0 - 1.0, default: 0.9)
- **fit_mode**: How to fit the fill image into the detected box:
  - `fit_height`: Scale to fit the height of the box (maintains aspect ratio)
  - `fit_width`: Scale to fit the width of the box (maintains aspect ratio)
  - `stretch`: Stretch to fill the entire box (may distort aspect ratio)
- **margin_size**: White margin in pixels to add around the fill image (0-50, default: 5)

### Output

- **output_image**: The base image with the fill image placed inside the detected white box

## How It Works

1. **White Box Detection**: The node converts the base image to grayscale and creates a binary mask to identify white areas above the specified threshold

2. **Contour Analysis**: Uses OpenCV to find contours and identifies the largest rectangular white area

3. **Margin Addition**: Adds a white margin around the fill image based on the margin_size parameter

4. **Image Fitting**: Resizes the fill image (with margin) according to the selected fit mode:
   - For `fit_height`: Scales to match box height, then checks if width fits
   - For `fit_width`: Scales to match box width, then checks if height fits
   - For `stretch`: Directly resizes to exact box dimensions

5. **Placement**: Centers the fitted image within the detected white box area

## Tips

- **White Threshold**: Lower values (0.7-0.8) detect slightly off-white areas, higher values (0.95+) only detect pure white
- **Fit Mode**: Use `fit_height` when you want to prioritize filling the vertical space, `fit_width` for horizontal space
- **Margin Size**: Set to 0 to disable margin, or increase for more white space around your content
- **Image Quality**: The node uses LANCZOS resampling for high-quality image resizing

## Requirements

- ComfyUI
- OpenCV (cv2)
- PIL (Pillow)
- NumPy
- PyTorch

## Example Workflow

1. Load a base image with a white rectangular area
2. Load the image you want to place inside the white area
3. Connect both to the Snaps Mask Fill node
4. Adjust the white threshold if needed
5. Choose your preferred fit mode
6. The output will show the base image with your content fitted inside the white box

## Troubleshooting

- **No white area detected**: Try lowering the white_threshold value
- **Wrong area detected**: Ensure the white box is the largest white area in the image
- **Image appears distorted**: Try different fit modes or check your input image aspect ratios