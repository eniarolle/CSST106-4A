# Exercise 1: HOG (Histogram of Oriented Gradients) Object Detection
```python
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt

image = cv2.imread('/content/butter.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



hog_features, hog_image = hog(
    gray_image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True,
    feature_vector=True,
)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(1, 2, 2)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Image')

plt.show()
```
![{7CD5E938-841F-4BEF-BB61-510C8E18A152}](https://github.com/user-attachments/assets/688ae6ea-a0b2-47f6-b3a9-aec7d3818148)
# Conclusion
One of the effective techniques used for the object detection technique is known as HOG, which refers to Histogram of Oriented Gradients. It pretty much applies to pedestrians in the given image as well and captures the gradient orientation distribution in small connected regions of the image to describe objects more explicitly about shape and appearance. Being robust towards light variations and small distortions, this method has excellent capabilities to highlight edge structures and patterns.

Although successful in so many applications, HOG performance is also dependent on the size and orientation of objects to be detected, among others. Also, careful parameter tuning for optimal results, such as cell size and block normalization, may be required. Despite these challenges, HOG remains one of the most widely used methods in computer vision today because of its balance of simplicity and effectiveness. Specially, as mentioned in previous work, HOG is often used as a basic method to design more complex techniques and models for object detection tasks. Therefore, HOG remains to be an excellent option for researchers and practitioners who are looking to implement reliable object detection systems.
# Exercise 2: YOLO (You Only Look Once) Object Detection
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO model
net = cv2.dnn.readNet("/content/yolov3.weights", "/content/yolov3.cfg")

# Load the class names (COCO dataset)
with open("/content/coco.names", "r") as f:
  classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# List of image file names
image_files = ["/content/three.jpg"]

processed_images = []
target_aspect_ratio = (3,2)  # 3:2 aspect ratio

def crop_to_aspect_ratio(image, aspect_ratio):
  """Crops the image to the desired aspect ratio by performing a center crop."""
  h, w, _ = image.shape
  target_w = w
  target_h = int(w / aspect_ratio[0] * aspect_ratio[1])

  # If the height is larger than the target height, we crop it vertically
  if target_h > h:
    target_h = h
    target_w = int(h * aspect_ratio[0] / aspect_ratio[1])

  # Compute cropping margins for center cropping
  x_margin = (w - target_w) // 2
  y_margin = (h - target_h) // 2

  cropped_image = image[y_margin:y_margin + target_h, x_margin:x_margin + target_w]
  return cropped_image

for image_file in image_files:
  # Load the image
  image = cv2.imread(image_file)

  # Crop the image to a 3:2 aspect ratio
  cropped_image = crop_to_aspect_ratio(image, target_aspect_ratio)
  height, width, channels = cropped_image.shape

  # Prepare the image for YOLO
  blob = cv2.dnn.blobFromImage(cropped_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
  net.setInput(blob)
  outs = net.forward(output_layers)

  # Process YOLO outputs
  class_ids = []
  confidences = []
  boxes = []

  for out in outs:
    for detection in out:
      scores = detection[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]
      if confidence > 0.5:
        # Object detected
        center_x = int(detection[0] * width)
        center_y = int(detection[1] * height)
        w = int(detection[2] * width)
        h = int(detection[3] * height)
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)

        boxes.append([x, y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)

  # Apply non-max suppression to remove overlapping boxes
  indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

  # Draw bounding boxes and labels with larger green font
  for i in indices:
    box = boxes[i]
    x, y, w, h = box
    label = str(classes[class_ids[i]])
    confidence = confidences[i]

    # Draw bounding box
    cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Increase font size and change color to green
    font_scale = 1.4  # Increase font size
    font_color = (0, 255, 0)  # Green color
    thickness = 2  # Thickness of the font
    cv2.putText(cropped_image, f"{label} {confidence:.2f}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)

 # Resize the image to a fixed width while maintaining aspect ratio
  desired_width = 640  # Choose your desired width
  aspect_ratio = cropped_image.shape[1] / cropped_image.shape[0] # Calculate aspect ratio
  desired_height = int(desired_width / aspect_ratio)  # Calculate desired height
  resized_image = cv2.resize(cropped_image, (desired_width, desired_height))

  # Append the resized image to the list
  processed_images.append(resized_image)

# Concatenate images vertically in a 5x1 grid
result_image = np.vstack(processed_images)

# Show the final image grid
plt.figure(figsize=(10, 30))
plt.imshow(result_image)
plt.axis('off')
plt.show()
```
![{609ED84A-D730-496C-8F8E-B9DF27E7986E}](https://github.com/user-attachments/assets/5f7ea039-f0fa-47a3-9a71-bae54973cb3b)
# Exercise 3: SSD (Single Shot MultiBox Detector) with TensorFlow
```python
!wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
!tar -zxf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
```
```python
import tensorflow as tf
import cv2
from google.colab.patches import cv2_imshow

# Load pre-trained SSD model
model = tf.saved_model.load('/content/ssd_mobilenet_v2_coco_2018_03_29/saved_model')

# Load the 'serving_default' signature
detection_fn = model.signatures['serving_default']

# Load image
image_path = '/content/three.jpg'
image_np = cv2.imread(image_path)

# Convert the image to a tensor and add a batch dimension
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]

# Run the detection function
detections = detection_fn(input_tensor)

# Extract detection details
detection_boxes = detections['detection_boxes'][0].numpy()
detection_scores = detections['detection_scores'][0].numpy()

# Visualize the bounding boxes
for i in range(detection_boxes.shape[0]):
    if detection_scores[i] > 0.5:
        ymin, xmin, ymax, xmax = detection_boxes[i]
        (left, right, top, bottom) = (xmin * image_np.shape[1], xmax * image_np.shape[1],
                                      ymin * image_np.shape[0], ymax * image_np.shape[0])

        # Draw bounding box on the image
        cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)

# Display the image with bounding boxes
cv2_imshow(image_np)
```
![{1592A992-DB1B-49DC-B5C5-9F4D633A11F3}](https://github.com/user-attachments/assets/64d0e6da-d7cf-4a5d-b293-646c9be64d04)
# Exercise 4: Traditional vs. Deep Learning Object Detection Comparison
```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load pre-trained SVM model
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load image
image = cv2.imread('/content/three.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform HOG-SVM based object detection
(rects, weights) = hog.detectMultiScale(gray_image, winStride=(8, 8), padding=(16, 16), scale=1.05)

# Draw bounding boxes for detected objects
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the output using cv2_imshow
cv2_imshow(image)


# Load YOLO model and configuration
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load an image
image = cv2.imread('/content/three.jpg')
height, width, channels = image.shape

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Process detections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Get bounding box coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image using cv2_imshow
cv2_imshow(image)
```
![{D03D9CCA-97D8-4481-AF33-C3A76B38A91E}](https://github.com/user-attachments/assets/406a13fd-50d1-4504-8ab8-afbba7544bee)
![{9341C917-51DB-43B5-B0CD-FD24E03B4814}](https://github.com/user-attachments/assets/294aecf1-d265-4b68-b4d2-8260d3aa6747)
