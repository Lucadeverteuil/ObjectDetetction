from ultralytics import YOLO 


'''
This class is Made to train the model However, I trained the model on Google Colab to better utilize the GPU and RAM
'''
#load new  model 
#uses the YOLOv8n model as a refernece 
model = YOLO("yolov8n.yaml")

#use model to detect objects
results = model.train(data = "NAME_OF_FILE.yaml",epochs = 100);  # train the model 

