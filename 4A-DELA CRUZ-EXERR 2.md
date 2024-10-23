# Task 1: SIFT Feature Extraction
```python
import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('/content/Buttercup_PPG_29.webp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints on the image
img_sift = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with keypoints
plt.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
plt.title("SIFT Keypoints")
plt.show()
```
![{ECD24647-F9E2-4E18-8276-22C0EA4AB7B3}](https://github.com/user-attachments/assets/e7dbc879-8dea-4587-b9df-698209d01b6a)
# Task 2: SURF Feature Extraction
```python
!apt-get update
!apt-get install -y cmake build-essential pkg-config

!git clone https://github.com/opencv/opencv.git
!git clone https://github.com/opencv/opencv_contrib.git

!mkdir -p opencv/build
%cd opencv/build
!cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D BUILD_EXAMPLES=OFF ..
!make -j8
!make install
```
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("/content/Buttercup_PPG_29.webp")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize SURF detector
surf = cv2.xfeatures2d.SURF_create()

# Detect keypoints and descriptors
keypoints, descriptors = surf.detectAndCompute(gray_image, None)

# Draw keypoints
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with keypoints
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title("SURF Keypoints")
plt.axis('off')
plt.show()
```
# Task 3: ORB Feature Extraction
```python
import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('/content/Buttercup_PPG_29.webp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and descriptors
keypoints, descriptors = orb.detectAndCompute(gray, None)

# Draw keypoints on the image
img_orb = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with keypoints
plt.imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
plt.title("ORB Keypoints")
plt.show()
```
![{0CCE2488-DE43-4CFD-9582-20FA0C4588B7}](https://github.com/user-attachments/assets/97245830-20fc-4740-af78-124ac336f110)

# Task 4: Feature Matching
```python
import cv2
import matplotlib.pyplot as plt

# Load two images
img1 = cv2.imread('/content/Buttercup_PPG_29.webp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/content/butter.png', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw the matches
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matched keypoints
plt.imshow(img_matches)
plt.title("Feature Matching (SIFT)")
plt.show()
```
![{FC278084-3DA4-4C46-98E4-10B7DB33D552}](https://github.com/user-attachments/assets/75283b4a-e8cb-4239-a9ef-c56fe9579bd3)

# Task5: Application of Feature Matching
```python

import numpy as np


# Load two images
img1 = cv2.imread('/content/Buttercup_PPG_29.webp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/content/butter.png', cv2.IMREAD_GRAYSCALE)


# Detect keypoints and descriptors using SIFT
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Match features using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test ( Love's ratio test)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract location of good matches
src_pts = np.float32(
    [keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

dst_pts = np.float32(
    [keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Find homography matrix
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp one image to align with the other
h, w = img1.shape[:2]
aligned_img = cv2.warpPerspective(img2, M, (w, h))

# Displat the result
plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
plt.title("Image Alignment using Homography")
plt.show()
```
![{9AB93947-186D-4F89-AA94-C5CB79861624}](https://github.com/user-attachments/assets/9365b21b-98ba-4549-af07-2db74ce8c3f5)

# Task 6: Combining Feature Extraction Methods
```python
# Load two images
image1 = cv2.imread('/content/Buttercup_PPG_29.webp', 0)
image2 = cv2.imread('/content/butter.png', 0)

# SIFT detector
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# ORB detector
orb = cv2.ORB_create()
keypoints1_orb, descriptors1_orb = orb.detectAndCompute(image1, None)
keypoints2_orb, descriptors2_orb = orb.detectAndCompute(image2, None)

# Initialize the Matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors for SIFT
matches_sift = bf.match(descriptors1_sift, descriptors2_sift)

# Use a different matcher for ORB (since ORB uses binary descriptors)
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf_orb.match(descriptors1_orb, descriptors2_orb)

# Sort matches by distance (Best matches first)
matches_sift = sorted(matches_sift, key=lambda x: x.distance)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)

# Draw the top matches from both methods
image_matches_sift = cv2.drawMatches(image1_resized, keypoints1_sift, image2_resized, keypoints2_sift, matches_sift[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
image_matches_orb = cv2.drawMatches(image1_resized, keypoints1_orb, image2_resized, keypoints2_orb, matches_orb[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Combine the two results into one for visualization
image_combined = np.hstack((image_matches_sift, image_matches_orb))

# Display the combined image
plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_BGR2RGB))
plt.title("SIFT (left) vs ORB (right)")
plt.axis('off')
plt.show()

# Print the number of matches for each method
print(f"Number of SIFT matches: {len(matches_sift)}")
print(f"Number of ORB matches: {len(matches_orb)}")
```
![{748B6E8B-EC7C-4C66-AE0B-949A171328ED}](https://github.com/user-attachments/assets/0499ad1d-74ab-4108-a35e-5309f8532aff)

# Overall Understanding
 SIFT is accurate for detecting and matching keypoints even in transformed images (scaled, rotated). SURF is faster than SIFT but still effective for keypoint detection. ORB is highly efficient and suited for real-time applications. Feature Matching is essential in comparing different images to find common objects or align them. Homography is used in aligning images, such as stitching images together to form a panorama
