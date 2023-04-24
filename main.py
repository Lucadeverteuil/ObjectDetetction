from ultralytics import YOLO 

#load new  model 
model = YOLO("yolov8n.yaml")

#use model to detect objects
results = model.train(data = "data.yaml",epochs = 100);  # train the model 

