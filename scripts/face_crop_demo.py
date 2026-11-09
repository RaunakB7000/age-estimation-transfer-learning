"""Detect a face with MediaPipe, expand the box by 25%, crop and resize to 224x224.

This is the same crop used by notebooks/01_data_preprocessing.ipynb, shown on a single image.
    python scripts/face_crop_demo.py path/to/image.jpg
"""
import sys

import cv2
import mediapipe as mp

if len(sys.argv) != 2:
    sys.exit(__doc__)

image = cv2.imread(sys.argv[1])
if image is None:
    sys.exit(f"Could not read image: {sys.argv[1]}")

detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
if not results.detections:
    sys.exit("No face detected!")

# Bounding box from MediaPipe (relative coordinates -> pixels)
bbox = results.detections[0].location_data.relative_bounding_box
h, w, _ = image.shape
x, y = int(bbox.xmin * w), int(bbox.ymin * h)
w_box, h_box = int(bbox.width * w), int(bbox.height * h)

# Expand by 25% on each side (value chosen by the team after testing), then clamp to the image
pad = 0.25
x -= int(w_box * pad)
y -= int(h_box * pad)
w_box += int(2 * w_box * pad)
h_box += int(2 * h_box * pad)
x, y = max(0, x), max(0, y)
w_box, h_box = min(w_box, w - x), min(h_box, h - y)

crop = cv2.resize(image[y:y + h_box, x:x + w_box], (224, 224))
cv2.imshow("Crop", crop)
cv2.waitKey(0)
