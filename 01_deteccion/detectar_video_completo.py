from ultralytics import YOLO
import cv2
import pandas as pd

# =========================
# RUTAS
# =========================
video_path = "/media/disk2tb/DosT/javier/MBG-V2/criaderos_mosquitos_v9/videos/video12_regionF.avi"
weights_path = "/media/disk2tb/DosT/javier/scripts/runs/detect/train_yolov12n_v52/weights/best.pt"

# =========================
# CARGA MODELO Y VIDEO
# =========================
model = YOLO(weights_path)
cap = cv2.VideoCapture(video_path)

frame_id = 0
detections = []  # aquí guardamos todo

# =========================
# LOOP FRAME POR FRAME
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Inferencia (una imagen = un frame)
    results = model.predict(
        source=frame,
        conf=0.5,
        iou=0.7,
        device="1",
        verbose=False
    )

    # =========================
    # PROCESAR RESULTADOS
    # =========================
    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            cls_name = model.names[cls_id]
            conf = float(boxes.conf[i].item())

            # Bounding box normalizada (YOLO format)
            x_center, y_center, bw, bh = boxes.xywhn[i].tolist()

            detections.append({
                "frame": frame_id,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "x_center_norm": x_center,
                "y_center_norm": y_center,
                "width_norm": bw,
                "height_norm": bh
            })

    frame_id += 1

    if frame_id % 100 == 0:       
        print(f"Frame {frame_id}", flush=True)

cap.release()

# =========================
# TABLA FINAL
# =========================
df = pd.DataFrame(detections)
print(df.head())

# Guardar a CSV si quieres
df.to_csv("detecciones_video12F_completo.csv", index=False)
