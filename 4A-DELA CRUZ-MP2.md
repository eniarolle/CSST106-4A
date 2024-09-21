# Hands-On Exploration:
# Lab Session 1: Image Trasformations
- Scaling and Rotation: Learn how to apply scaling and rotation transformations to images using OpenCV.
- Implementation: Practice these transformations on sample images provided in the lab.

```python
import matplotlib.pyplot as plt
# Plotting the images in a single row
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Display the original image
axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")

# Display the scaled image
axes[1].imshow(resized_image)
axes[1].set_title("Scaled Image (50%)")

# Display the rotated image
axes[2].imshow(rotated_image)
axes[2].set_title(f"Rotated Image ({angle}°)")

# Adjust the layout and display all images
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/3b878633-8404-418a-8daa-51b1882c49de)

# Lab Session 2: Filtering Techniques
- Blurring and Edge Detection: Explore how to apply blurring filters and edge detection algorithms to images using OpenCV.
- Implementation: Apply these filters to sample images to understand their effects.
``` python
# Applying Gaussian, Median, and Bilateral filters
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
median_blur = cv2.medianBlur(image, 5)
bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)

# Apply Canny edge detection to original and blurred images
edges = {
    "Original": cv2.Canny(image_rgb, 100, 200),
    "Gaussian Blur": cv2.Canny(gaussian_blur, 100, 200),
    "Median Blur": cv2.Canny(median_blur, 100, 200),
    "Bilateral Blur": cv2.Canny(bilateral_filter, 100, 200)
}

# Create 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Loop through the edge-detected images for cleaner plotting
for ax, (title, edge_img) in zip(axes.ravel(), edges.items()):
    ax.imshow(edge_img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')  # Hide axes for cleaner display

# Adjust layout and show the images
plt.tight_layout()
plt.show()
plt.show()
```

![image](https://github.com/user-attachments/assets/9ab41407-37d8-4fc9-b6d9-0d9b3da3b499)

# Problem-Solving Session:
- Common Image Processing Tasks:
 - Engage in a problem-solving session focused on common challenges encountered in image processing tasks.
 - Scenario-Based Problems: Solve scenarios where you must choose and apply appropriate image processing techniques.

# Image Enhancement:
# Scenario: The image is too dark or lacks contrast. How can you enhance the visual quality?

# Techniques: Histogram Equalization:
- Increases contrast by redistributing pixel intensity values.
- CLAHE (Contrast Limited Adaptive Histogram Equalization): Enhances contrast in localized areas.
```python
# Convert to grayscale for histogram equalization
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Histogram Equalization
equalized_image = cv2.equalizeHist(gray_image)

# Display the result
plt.imshow(equalized_image, cmap='gray')
```
# Color Space Conversion:
# Scenario: You need to convert an image to grayscale or another color space for further analysis.

# Techniques:
- Grayscale Conversion: Reduces the image to a single intensity channel.
- HSV Conversion: Useful for separating color information from intensity.
  ``` python
  # Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert the image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Display the grayscale image
plt.imshow(gray_image, cmap='gray')
```
# Object Detection:
# Scenario: You need to detect and locate specific objects in an image (e.g., faces, cars).

# Techniques:
- Template Matching: Search for a template image in a larger image.
- Contour Detection: Detects the outlines of objects based on intensity changes.
```python
# Detect contours in the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Display the result
plt.imshow(image)
```
# Assignment:
# Implementing Image Transformations and Filtering:
- Choose a set of images and apply the techniques you've learned, including scaling, rotation, blurring, and edge detection.
- Documentation: Document the steps taken, and the results achieved in a report.

```python
image = cv2.imread('/content/DELACRUZ (1).jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

scale_percent = 50
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(image, dim)

angle = 45
center = (image.shape[1] // 2, image.shape[0] // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

# Gaussian Blur
Gaussian = cv2.GaussianBlur(image, (5, 5), 0)
# Median Blur
median = cv2.medianBlur(image, 5)
# Bilateral Blur
bilateral = cv2.bilateralFilter(image, 9, 75, 75)

fig, axes = plt.subplots(2, 3, figsize=(10, 5))

axes[0][0].imshow(image)
axes[0][0].set_title("Original Image")

axes[0][1].imshow(resized_image)
axes[0][1].set_title("Scaled Image")

axes[0][2].imshow(rotated_image)
axes[0][2].set_title("Rotated Image")

axes[1][0].imshow(Gaussian)
axes[1][0].set_title("Gaussian Filter")

axes[1][1].imshow(median)
axes[1][1].set_title("Median Filter")

axes[1][2].imshow(bilateral)
axes[1][2].set_title("Bilateral Filter")


plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/5cb16cb6-abd3-4f65-970b-2f24c9b2267e)

Machine Problem No. 2:
We used the library OpenCV to apply various types of image processing techniques. We have used filtering techniques, which include Gaussian, Median, and Bilateral Blur, as well as picture transformations, such as rotation and scaling. We tried to understand the methods and use them to improve and modify the pictures.
# Image Transformations
# Objective:
To scale and rotate an image.
Steps:
- Scaling: Reduced the image size to 50% .
- Rotation: Rotated the image by 45 degrees.

# Filtering Techniques
# Objective:
To apply different blurring filters to the image.
Steps:
- Gaussian Blur: Applied using a 5x5 kernel.
- Median Blur: Applied to reduce noise.
- Bilateral Filter: Applied to reduce noise while keeping edges sharp.
# Conclusion
- Scaling and rotation were successfully applied to modify the image’s size and orientation.
- Blurring filters helped reduce noise, with the bilateral filter providing the best balance between noise reduction and edge preservation.
These techniques are useful for various image processing tasks and were successfully demonstrated in this report.
