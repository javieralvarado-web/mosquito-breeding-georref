from ultralytics import YOLO

dir_weights = "/media/disk2tb/DosT/javier/scripts/runs/detect/train_yolov12n_v52/weights/best.pt"
dir_directions = "/media/disk2tb/DosT/javier/MBG-V2/train_yolov12_v5.yaml"

# Load a model
model = YOLO(dir_weights)

# Customize validation settings
validation_results = model.val(data=dir_directions, imgsz=1024, batch=1, conf = 0.5, iou = 0.7, device="1", split='val',
name = 'valid_yolov12n_v5', plots=True, save_json=True)