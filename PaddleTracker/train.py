from ultralytics import YOLO

# Load a model
model = YOLO("models/yolov8s-oiv7.pt") 

# Train the model
results = model.train(data="paddle_images/paddle_images.yaml", epochs=100, imgsz=640)