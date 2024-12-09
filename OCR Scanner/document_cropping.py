import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load image using OpenCV."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image

def preprocess_image(image):
    """Convert image to grayscale and apply Gaussian blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred

def detect_edges_with_morph(image):
    """Detect edges using Canny edge detection and apply morphological operations."""
    edges = cv2.Canny(image, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed

def debug_display_contours(image, contours):
    """Display the detected contours for debugging purposes."""
    debug_image = image.copy()
    cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Contours")
    plt.show()

def find_document_contour_filtered(edges, image):
    """Find the largest contour that resembles a document, with size filtering."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    debug_display_contours(image, contours)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx

    raise ValueError("Document contour not found.")

def order_points(pts):
    """Order points in the order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_perspective(image, points):
    """Perform a perspective transform to straighten the document."""
    rect = order_points(points.reshape(4, 2))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def crop_image(image, crop_size=20):
    """Crop the image by a specified number of pixels on each side."""
    h, w = image.shape[:2]
    return image[crop_size:h - crop_size, crop_size:w - crop_size]

def apply_adaptive_threshold(image):
    """Apply adaptive threshold to the image with improved parameters."""
    adaptive_thresh = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        5
    )
    return adaptive_thresh

def additional_processing(image):
    """Apply additional processing: adaptive threshold and median blur."""
    adaptive_thresh = apply_adaptive_threshold(image)
    blurred = cv2.medianBlur(adaptive_thresh, 3)
    return blurred


def process_document_with_improved_contours(image_path, output_path):
    """Main function with improved contour filtering and additional processing."""
    image = load_image(image_path)
    gray, blurred = preprocess_image(image)
    edges = detect_edges_with_morph(blurred)

    plt.imshow(edges, cmap='gray')
    plt.title("Edges After Morphological Operations")
    plt.show()

    try:
        doc_contour = find_document_contour_filtered(edges, image)
    except ValueError as e:
        print(f"Error: {e}")
        return

    straightened = warp_perspective(image, doc_contour)
    cropped = crop_image(straightened)
    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    processed = additional_processing(cropped_gray)

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.title("Straightened and Cropped")
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 3)
    plt.title("Final Processed Image")
    plt.imshow(processed, cmap='gray')
    plt.show()

    cv2.imwrite(output_path, processed)
    print(f"Processed image saved to: {output_path}")

    return processed

if __name__ == "__main__":
    input_image_path = "123456.jpg"
    output_image_path = "processed_image.jpg"
    processed_image = process_document_with_improved_contours(input_image_path, output_image_path)
