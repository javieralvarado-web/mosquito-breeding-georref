from ultralytics import YOLO

dir_weights = "/media/disk2tb/DosT/javier/weights/yolo12n.pt"
dir_directions = "/media/disk2tb/DosT/javier/MBG-V2/train_yolov12_v5.yaml"

# Load a model
#model = YOLO(dir_conf_model)  # construye un nuevo modelo a partir de yaml
model = YOLO(dir_weights)  # se cargan los pesos preentrenados de otro modelo 
#model = YOLO(dir_conf_model).load(dic_weights)  # construye el modelo transfiriendo los pesos cargados 

# Train the model
model.train(data=dir_directions, epochs=200, imgsz=1024, batch=8, device='1', patience = 400, name = 'train_yolov12n_v5')