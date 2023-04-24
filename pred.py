from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0)
cap.set(3,700)
cap.set(4,720)

model = YOLO('best.pt')
classNames = ["TolitPaper", "hole"]
while True:  
    success, img = cap.read()
    results = model(img, stream = True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
    