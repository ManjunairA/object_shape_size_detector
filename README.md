# Image shape detector

This project is a simple image processing script that analyzes an image to find contours, highlight edges and corners, and calculate/display edge lengths. The script uses OpenCV and NumPy libraries for image processing.

## Prerequisites

Before you begin, ensure you have the following libraries installed:
- Python : 3.9.13
- OpenCV : 4.8.1.78
- NumPy : 1.26..4

You can install them using pip:

```bash
pip install opencv-python
pip install numpy
```
OR
```bash
pip install -r requirements.txt
```

## Script Description
The script performs the following steps:

1. **Load the Image**: Reads an image from the specified file path.

2. **Preprocess the Image**: Converts the image to grayscale and applies thresholding to create a binary image.

3. **Find Contours**: Detects contours in the binary image.

4. **Highlight Edges and Corners**: Draws edges with green lines, highlights corners with red circles, and calculates/displays edge lengths.

5. **Save the Output Image**: Saves the processed image with highlighted edges and corners.

## Usage
To use the script, follow these steps:

1. Place your input image in the same directory as the script or provide the full path to the image.

2. Replace the image_path variable in the ***main*** function with the path to your input image.

3. Run the script:

```bash
python shape_detector.py
```