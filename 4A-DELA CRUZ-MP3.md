# Machine Problem No. 3: Feature Extraction and Object Detection
# Objective:
The objective of this machine problem is to implement and compare the three feature extraction methods
(SIFT, SURF, and ORB) in a single task. You will use these methods for feature matching between two
images, then perform image alignment using homography to warp one image onto the other.
# Step 1: Load Images
```python
import cv2
import matplotlib.pyplot as plt

# Load two images that depict the same scene or object from different angles
image1 = cv2.imread('/content/butter.png', 0)  # Load the first image in grayscale
src = cv2.imread('/content/Buttercup_PPG_29.webp', 0)  # Load the second image in grayscale
image2 = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
# Check if images are loaded correctly

plt.figure(figsize=(10, 5))

# Show first image
plt.subplot(1, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title('Image 1')
plt.axis('off')

# Show second image
plt.subplot(1, 2, 2)
plt.imshow(image2, cmap='gray')
plt.title('Image 2')
plt.axis('off')

plt.tight_layout()
plt.show()
```
![{9CE0AB54-92E7-4361-8C6C-DAE001119B7F}](https://github.com/user-attachments/assets/a63746fa-4356-4cbb-ac9f-02b3a13836d3)

# Step 2: Extract Keypoints and Descriptors Using SIFT, SURF, and ORB (30 points)
- Apply the SIFT algorithm to detect keypoints and compute descriptors for both images.
- Apply the SURF algorithm to do the same.
- Finally, apply ORB to extract keypoints and descriptors.
```python
sift = cv2.SIFT_create()
keypoints1_sift, descriptors1_sift = sift.detectAndCompute(image1, None)
keypoints2_sift, descriptors2_sift = sift.detectAndCompute(image2, None)

surf = cv2.xfeatures2d.SURF_create()
keypoints1_surf, descriptors1_surf = surf.detectAndCompute(image1, None)
keypoints2_surf, descriptors2_surf = surf.detectAndCompute(image2, None)

orb = cv2.ORB_create()
keypoints1_orb, descriptors1_orb = orb.detectAndCompute(image1, None)
keypoints2_orb, descriptors2_orb = orb.detectAndCompute(image2, None)

image1_sift_keypoints = cv2.drawKeypoints(image1, keypoints1_sift, None, color=(255, 0, 0))
image2_sift_keypoints = cv2.drawKeypoints(image2, keypoints2_sift, None, color=(255, 0, 0))

image1_surf_keypoints = cv2.drawKeypoints(image1, keypoints1_surf, None, color=(0, 255, 0))
image2_surf_keypoints = cv2.drawKeypoints(image2, keypoints2_surf, None, color=(0, 255, 0))

image1_orb_keypoints = cv2.drawKeypoints(image1, keypoints1_orb, None, color=(0, 0, 255))
image2_orb_keypoints = cv2.drawKeypoints(image2, keypoints2_orb, None, color=(0, 0, 255))

plt.figure(figsize=(15, 10))

plt.subplot(3, 2, 1)
plt.imshow(cv2.cvtColor(image1_sift_keypoints, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoints (Image 1)')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(cv2.cvtColor(image2_sift_keypoints, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoints (Image 2)')
plt.axis('off')

plt.subplot(3, 2, 3)
plt.imshow(cv2.cvtColor(image1_surf_keypoints, cv2.COLOR_BGR2RGB))
plt.title('SURF Keypoints (Image 1)')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(cv2.cvtColor(image2_surf_keypoints, cv2.COLOR_BGR2RGB))
plt.title('SURF Keypoints (Image 2)')
plt.axis('off')

plt.subplot(3, 2, 5)
plt.imshow(cv2.cvtColor(image1_orb_keypoints, cv2.COLOR_BGR2RGB))
plt.title('ORB Keypoints (Image 1)')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.imshow(cv2.cvtColor(image2_orb_keypoints, cv2.COLOR_BGR2RGB))
plt.title('ORB Keypoints (Image 2)')
plt.axis('off')

plt.tight_layout()
plt.show()
```
![{FF488D32-F6A0-4E6D-9F71-BBB4D27C7BAF}](https://github.com/user-attachments/assets/53e22bd5-f054-4825-81e7-cdeee134bddc)

#  Feature Matching with Brute-Force and FLANN (30 points)
- Match the descriptors between the two images using Brute-Force Matcher.
- Repeat the process using the FLANN Matcher.
- For each matching method, display the matches with lines connecting corresponding keypoints
between the two images.
```python
# Brute-Force Matcher with SIFT
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) 
matches_bf = bf.match(descriptors1_sift, descriptors2_sift)
matches_bf = sorted(matches_bf, key=lambda x: x.distance)  # Sort by distance

# FLANN Matcher with SIFT
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50) 

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches_flann = flann.knnMatch(descriptors1_sift, descriptors2_sift, k=2)

# Apply ratio test for FLANN matching (Lowe's ratio test)
good_matches_flann = []
for m, n in matches_flann:
    if m.distance < 0.7 * n.distance:
        good_matches_flann.append(m)

# Draw the matches for Brute-Force
image_bf_matches = cv2.drawMatches(image1, keypoints1_sift, image2, keypoints2_sift, matches_bf[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Draw the matches for FLANN
image_flann_matches = cv2.drawMatches(image1, keypoints1_sift, image2, keypoints2_sift, good_matches_flann[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the results
plt.figure(figsize=(15, 10))

# Brute-Force Matches
plt.subplot(1, 2, 1)
plt.imshow(image_bf_matches)
plt.title('Brute-Force Matches (SIFT)')
plt.axis('off')

# FLANN Matches
plt.subplot(1, 2, 2)
plt.imshow(image_flann_matches)
plt.title('FLANN Matches (SIFT)')
plt.axis('off')

plt.tight_layout()
plt.show()
```
![{9023FC3F-67AE-4D43-9067-F6D0B5336BAB}](https://github.com/user-attachments/assets/4f6dc9be-faed-4a04-bcec-f8c14eb38164)

```python
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_bf = bf.match(descriptors1_orb, descriptors2_orb)
matches_bf = sorted(matches_bf, key=lambda x: x.distance)

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches_flann = flann.knnMatch(descriptors1_orb, descriptors2_orb, k=2)

good_matches_flann = []
for match in matches_flann:
    if len(match) == 2:
        m, n = match
        if m.distance < 0.7 * n.distance:
            good_matches_flann.append(m)

image_bf_matches = cv2.drawMatches(image1, keypoints1_orb, image2, keypoints2_orb, matches_bf[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

image_flann_matches = cv2.drawMatches(image1, keypoints1_orb, image2, keypoints2_orb, good_matches_flann[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(15, 10))

plt.subplot(1, 2, 1)
plt.imshow(image_bf_matches)
plt.title('Brute-Force Matches (ORB)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_flann_matches)
plt.title('FLANN Matches (ORB)')
plt.axis('off')

plt.tight_layout()
plt.show()
```
![{FD1FB0B1-F210-4024-9FFB-F21A714C8D0C}](https://github.com/user-attachments/assets/04a9a481-5da4-448e-997a-ae238648fc3a)

# Step 4: Image Alignment Using Homography (20 points)
- Use the matched keypoints from SIFT (or any other method) to compute a homography matrix.
- Use this matrix to warp one image onto the other.
- Display and save the aligned and warped images.
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread('/content/butter.png')
image2 = cv2.imread('/content/Buttercup_PPG_29.webp')

gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1,None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2,None)

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(descriptors1,descriptors2, k=2)

good_matches =[]
for m,n in matches:
  if m.distance < 0.75* n.distance:
    good_matches.append(m)

# Ensure src_pts and dst_pts are derived from 'good_matches' and corresponding keypoints
# Use keypoints1 and keypoints2 instead of keypoints1_orb and keypoints2_orb
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Enforce correct data type and dimensions for homography matrix
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
if M is not None:
    M = M.astype(np.float32)  # Ensure M is of type float32
else:
    print("Homography matrix could not be calculated.")
    # Handle the case where findHomography fails to find a good transformation

h, w, _ = image1.shape

result = cv2.warpPerspective(image1, M, (w, h))

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Image Alignment Using Homography")
plt.show()
```
![{34C3F464-F30D-4148-A5E7-33CD996DBB21}](https://github.com/user-attachments/assets/82cad6f2-40e6-4d0b-9561-9c4222f1db4f)

# Step 5: Performance Analysis (20 points)
SIFT is highly accurate for detecting keypoints in complex images, but it requires significant computational resources. SURF, which stands for Speeded-Up Robust Features, is faster and better suited for real-time applications. ORB, which combines Oriented FAST and Rotated BRIEF, is currently the fastest algorithm, though it struggles with changes in scale and lighting, leading to reduced keypoint detection accuracy. SURF detects more keypoints than the others due to its use of the Hessian matrix in its detector, allowing it to match features more quickly in real-time scenarios. ORB is faster than both SIFT and SURF because it uses the FAST keypoint detector and the BRIEF descriptor. SIFT, however, is the slowest because of its complex calculations. When it comes to feature matching, Brute-Force Matcher and FLANN Matcher each have their strengths. For high-precision tasks, SIFT combined with Brute-Force Matcher provides the most accurate solution, while ORB with FLANN Matcher is the fastest and most efficient, though with some trade-offs in accuracy.


