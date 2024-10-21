# Exercise 1: Harris Corner Detection
# Task: Detect corners in an image using the Harris Corner Detection algorithm.
# Steps:
- Load an image.
- Convert it to grayscale.
- Apply the Harris Corner Detection method to detect corners.
- Visualize and display the corners.
# Key Points:
- Harris Corner Detection is used to find intersection points in object edges.
```python
  import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("/content/Buttercup_PPG_29.webp")

# Converting the image into grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Harris Corner Detection
harris_corners = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)

# Dilate the result to mark the corners
harris_corners = cv2.dilate(harris_corners, None)

# Threshold for an optimal value, it may vary depending on the image
threshold = 0.01 * harris_corners.max()

# Create a copy of the original image to draw corners on
image_with_corners = np.copy(image)

# Get the coordinates of detected corners
y, x = np.where(harris_corners > threshold)

# Iterate through all the corners and draw larger circles on the image
for i, j in zip(x, y):
    cv2.circle(image_with_corners, (i, j), radius=5, color=(0, 0, 255), thickness=2)

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
plt.title('Image with Detected Corners')
plt.axis('off')

plt.tight_layout()
plt.show()
```
![{693177AC-3B94-4F94-B74B-74A42D776F27}](https://github.com/user-attachments/assets/01082de8-b09f-489d-a4a0-52db9dea5acc)

# Exercise 2: HOG (Histogram of Oriented Gradients) Feature Extraction
# Task: Extract features using the HOG descriptor.
# Steps:
- Load an image of an object or person.
- Convert the image to grayscale.
- Apply the HOG descriptor to extract features.
- Visualize the gradient orientations on the image.
# Key Points:
- HOG captures object structures through gradient orientations.
```python
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import exposure

# Load img
img = cv2.imread('/content/Buttercup_PPG_29.webp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# HOG feature extraction
hog_features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')

# Rescale HOG img for better visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Display HOG
plt.imshow(hog_image_rescaled, cmap='gray')
plt.title('HOG Feature Extraction')
plt.show()
```
![{3284CE23-5F25-4CBA-8090-9B425D846EF6}](https://github.com/user-attachments/assets/328c1366-ce36-4c93-89a6-ae72042f4b53)

# Exercise 3: FAST (Features from Accelerated Segment Test) Keypoint Detection
# Task: Detect keypoints using the FAST algorithm.
# Steps:
- Load an image.
- Convert it to grayscale.
- Apply the FAST algorithm to detect keypoints.
- Visualize and display the keypoints.
# Key Points:
- FAST is a quick and efficient keypoint detector.
```python
import cv2
import matplotlib.pyplot as plt

# Load img
img = cv2.imread('/content/Buttercup_PPG_29.webp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# FAST keypoint detector
fast = cv2.FastFeatureDetector_create()
keypoints = fast.detect(gray, None)

# Draw keypoints
image_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0))

# Display result
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('FAST Keypoint Detection')
plt.show()
```
![{A889BE0E-8F40-4F20-9B4F-F3F1DF738A0F}](https://github.com/user-attachments/assets/50b4a854-be0f-4395-8f23-d7b4d709bd33)

# Exercise 4: Feature Matching using ORB and FLANN
# Task: Use ORB descriptors to find and match features between two images using FLANN-based matching.
- Load two images of your choice.
- Extract keypoints and descriptors using ORB.
- Match features between the two images using the FLANN matcher.
- Display the matched features.
# Key Points:
- ORB is fast and efficient, making it suitable for resource-constrained environments.
- FLANN (Fast Library for Approximate Nearest Neighbors) speeds up the matching process, making
it ideal for large datasets.
```python
import cv2
import matplotlib.pyplot as plt

# Load two images
img1 = cv2.imread('/content/Buttercup_PPG_29.webp')
img2 = cv2.imread('/content/butter.png')

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# FLANN-based matcher parameters
index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Draw matches
result_image = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)

# Display result
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title('ORB Feature Matching with FLANN')
plt.show()
```
![{D5F5F19E-4E6E-4DF7-A2F3-2127AFD6816B}](https://github.com/user-attachments/assets/f4d642e3-89c4-4940-a17d-0d0af5c7f8ac)

# Exercise 5: Image Segmentation using Watershed Algorithm
# Task: The Watershed algorithm segments an image into distinct regions.
- Load an image.
- Apply a threshold to convert the image to binary.
- Apply the Watershed algorithm to segment the image into regions.
- Visualize and display the segmented regions.
# Key Points:
- Image segmentation is crucial for object detection and recognition.
- The Watershed algorithm is especially useful for separating overlapping objects.
These exercises extend the concepts covered in the document, introducing a mix of foundational and realtime applications. Let me know if you'd like more details on any specific task!

```python
image = cv2.imread('/content/Buttercup_PPG_29.webp')
original = image.copy()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

sure_bg = cv2.dilate(opening, kernel, iterations=3)

sure_fg = np.uint8(sure_fg)
unknown_region = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown_region == 255] = 0

markers = cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]  

plt.figure(figsize=(15, 7))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Image Segmentation using Watershed Algorithm')
plt.axis('off')

plt.show()
```
![{00566900-9577-40C1-A349-B7D74463853F}](https://github.com/user-attachments/assets/83c6350e-2c4e-4577-b27a-9e78ac6eb616)

# Conclusion
The Watershed algorithm is a powerful image segmentation technique, particularly effective for distinguishing regions or objects in images where boundaries are not clearly defined. It treats the image as a topographical surface, with pixel intensities representing elevations. Segmentation is achieved by "flooding" the surface from predefined markers in areas of interest. When floods from different markers meet, watershed lines or boundaries form, dividing the image into distinct segments. As the algorithm progresses, the final segmentation result is produced.

One key advantage of the Watershed algorithm is its ability to accurately segment overlapping or touching objects, which simpler methods often struggle to handle. However, the algorithm can be highly sensitive to noise and can lead to over-segmentation if the initial markers or gradient definitions are not properly set. To mitigate this, noise reduction techniques or markers derived from other methods, like distance transforms or edge detection, are commonly used before applying the algorithm. Overall, when combined with these techniques, the Watershed algorithm proves to be a robust tool for precise image segmentation, especially in complex cases.



