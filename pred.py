from ultralytics import YOLO
import cv2
import cvzone 
import math

'''
This Project Uses a YOLOv8n model to detect PPE Equipent in a Video Stream Using Open CV.
This model on uses 100 epochs and was trained on a dataset of 1000 images, Would recommend training on more images and more epochs to get better results.
'''

#Set the camera and Resolution
cap = cv2.VideoCapture(0)
cap.set(3,1680)
cap.set(4,720)


#Delcare the model and the classnames
model = YOLO('Con2.pt')
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']
# Creating a list of colors for the bounding boxes
while True:  
    success, img = cap.read()
    results = model(img, stream = True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w,h, = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w ,h) )
            conf = math.ceil((box.conf[0] * 100)) /100
            cls = int(box.cls[0])
            #Display Box, Name and Confidence of the object
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1), max(35,y1)), thickness=2)
            
    cv2.imshow("Image",img)
    cv2.waitKey(1)


'''
This PPE-DETECTION Dataset is made available under the Public Domain Dedication and License v1.0 whose full text can be found at:  https://universe.roboflow.com/ai-project-yolo/ppe-detection-q897z 
This dataset was created by AI Project YOLO and originally sourced from Roboflow.
Its intentions was to show how easy it is to create a dataset and train a model using YOLO v8.
'''


# @misc{ ppe-detection-q897z_dataset,
#     title = { PPE-DETECTION Dataset },
#     type = { Open Source Dataset },
#     author = { Ai Project YOLO },
#     howpublished = { \url{ https://universe.roboflow.com/ai-project-yolo/ppe-detection-q897z } },
#     url = { https://universe.roboflow.com/ai-project-yolo/ppe-detection-q897z },
#     journal = { Roboflow Universe },
#     publisher = { Roboflow },
#     year = { 2023 },
#     month = { apr },
#     note = { visited on 2023-05-01 },
# }