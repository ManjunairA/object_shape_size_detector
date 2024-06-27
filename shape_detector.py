"""Shape detector and edge length analyser."""
import cv2
import numpy as np

def load_image(image_path):
    """
    Load an image from the specified file path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        image (numpy.ndarray): The loaded image.
    """
    return cv2.imread(image_path)

def preprocess_image(image):
    """
    Convert the image to grayscale and apply thresholding to create a binary image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        thresh (numpy.ndarray): The thresholded binary image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def find_contours(binary_image):
    """
    Find contours in the binary image.

    Args:
        binary_image (numpy.ndarray): The binary image.

    Returns:
        contours (list): A list of detected contours.
    """
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def highlight_edges_and_corners(image, contours):
    """
    Highlight edges with green lines, corners with red circles, and calculate/display edge lengths.

    Args:
        image (numpy.ndarray): The input image.
        contours (list): A list of contours to process.

    Returns:
        image (numpy.ndarray): The processed image with highlighted edges, corners, and edge lengths.
    """
    for contour in contours:
        # Approximate the contour to get edges and corners
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Draw edges with green lines
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        
        # Draw corners with small red circles
        for point in approx:
            cv2.circle(image, tuple(point[0]), 5, (0, 0, 255), -1)
        
        # Calculate and display the length of each straight edge
        for i in range(len(approx)):
            start_point = tuple(approx[i][0])
            end_point = tuple(approx[(i+1) % len(approx)][0])
            edge_length = int(np.linalg.norm(np.array(start_point) - np.array(end_point)))
            
            # Display the edge length
            midpoint = (int((start_point[0] + end_point[0]) / 2), int((start_point[1] + end_point[1]) / 2))
            cv2.putText(image, str(edge_length), midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
    return image

def main(image_path):
    """
    Main function to load an image, preprocess it, find contours, highlight edges and corners,
    and save the output image.

    Args:
        image_path (str): The path to the input image file.
    """
    # Load and preprocess the image
    image = load_image(image_path)
    binary_image = preprocess_image(image)
    
    # Find contours
    contours = find_contours(binary_image)
    
    # Number of shapes
    num_shapes = len(contours)
    print(f"Number of shapes: {num_shapes}")
    
    # Highlight edges and corners
    output_image = highlight_edges_and_corners(image.copy(), contours)
    
    # Save the result
    output_image_path = 'output_analysed_image.png'
    cv2.imwrite(output_image_path, output_image)
    print(f"The output image saved as {output_image_path}")

if __name__ == "__main__":
    image_path = 'img.png'  # Replace with the path to your input image
    main(image_path)
