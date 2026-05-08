from ultralytics import YOLO
import os

project_dir = "/content/drive/MyDrive/AI_detection"   
data_yaml = os.path.join(project_dir, "data", "data.yaml") 
weights_path = os.path.join(project_dir, "yolov12n.pt")     


model = YOLO(weights_path)

model.train(
    data=data_yaml,        
    epochs=30,            
    imgsz=640,             
    batch=16,             
    name="defect_yolov12n",
    project=os.path.join(project_dir, "runs"), 
    pretrained=True,       
    patience=20,          
    workers=2              
)

metrics = model.val()  
print("📊 Validation Metrics:")
print(metrics)
 
model.export(format="onnx")    