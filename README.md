SOURCE CODE: 
import torch from matplotlib import 
pyplot as plt import numpy as np 
import cv2 import requests from io 
import BytesIO import 
matplotlib.pyplot as plt import 
numpy as np from PIL import 
Image # Load the YOLOv5 model 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
img='https://thumbs.dreamstime.com/z/motorcycle-rider-riding-his-yamaha-yzfrsukoharjo-indonesia-february-yamaha-entry-level-sport-bike-yamaha-has246541895.jpg?w=576.jpg' results=model(img) 
class_indices = results.pred[0][:, -1].cpu().numpy() 
bounding_boxes = results.pred[0][:, :-
1].cpu().numpy() target_class_indices = [1, 2, 3, 5, 7] 
# Find the indices corresponding to the target classes tie_indices = 
np.where(np.isin(class_indices, target_class_indices))[0] 
# Find the indices corresponding to the "tie" class (class index 27) # 
tie_indices = np.where(class_indices == 27)[0] 
# Extract the bounding boxes and labels for the "tie" class 
tie_boxes = bounding_boxes[tie_indices] tie_labels 
= class_indices[tie_indices] 
# Assuming you have the image loaded as 'image' (you should replace this with your 
actual image data) # image = ... 
image_url = 'https://thumbs.dreamstime.com/z/motorcycle-rider-riding-hisyamahayzf-r-sukoharjo-indonesia-february-yamaha-entry-level-sport-bike-yamahahas246541895.jpg?w=576.jpg' 
# Download the image from the URL response 
= requests.get(image_url) image_bytes = 
BytesIO(response.content) image = 
np.array(Image.open(image_bytes)) 
# Create a copy of the image to draw bounding boxes 
image_with_bboxes = np.copy(image).astype(np.uint8) 
# Iterate over the "tie" bounding boxes and labels for i 
in range(len(tie_boxes)): 
 x1, y1, x2, y2, confidence = tie_boxes[i] # Bounding box coordinates and 
confidence score label = results.names[tie_labels[i]] # Class label 
 # Convert coordinates to integers x1, y1, x2, 
y2 = int(x1), int(y1), int(x2), int(y2) # Draw a 
green rectangle with confidence score 
cv2.rectangle(image_with_bboxes, (x1, y1), (x2, 
y2), (0, 255, 0), 2) # Green color: (0, 255, 0), Line 
thickness: 2 
 # Define text settings 
 font = cv2.FONT_HERSHEY_SIMPLEX 
 font_scale = 0.75 # Adjust the font size here font_thickness = 2 
text = f"{label}: {confidence:.2f}" # Format label and confidence 
 # Get text size to calculate text position 
 (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 
font_thickness) 
 
 # Calculate the posqition to place the text above the bounding box 
text_x = x1 text_y = y1 - 10 - text_height # Adjust the vertical 
position here 
 
 # Draw the green rectangle and text cv2.rectangle(image_with_bboxes, (x1, y1), 
(x2, y2), (0, 255, 0), 2) cv2.putText(image_with_bboxes, text, (text_x, text_y), 
font, font_scale, (0, 255, 0), font_thickness) 
# Display the image with bounding boxes and confidence scores 
cv2.imshow("Image with Bounding Boxes",image_with_bboxes) 
cv2.waitKey(0)cv2.destroyAllWindows()
