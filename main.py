from ultralytics import YOLO

print("🔥 Starting FireWatch AI - Made with 💗 by Azad Bhasme")

# Loading the custom YOLOv8 model
model = YOLO('best.pt')

# Run prediction using webcam,confidence level should be > 60,image size 
model.predict(source=0, imgsz=640, conf=0.6, save=True)

