import cv2
import numpy as np
from ultralytics import YOLO

# --------- USTAWIENIA ---------
STREAM_URL = 0  # 0 = domyślna kamera laptopa, zmień na strumień RC-czołgu
MODEL_WEIGHTS = "yolov8n.pt"
CLASSES_KEEP = ["person"]
MIN_CONF = 0.70

# Rozdzielczość kamery (Full HD)
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Kontury Canny
ALPHA_EDGES = 0.5  # przezroczystość konturów
KERNEL_SIZE = 2    # pogrubienie konturów

# --------- Przygotowanie modelu ---------
model = YOLO(MODEL_WEIGHTS)

# --------- Funkcja przetworzonego HUD ---------
def draw_hud(frame, boxes, alpha_edges=ALPHA_EDGES, kernel_size=KERNEL_SIZE):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 1. Kontury przeszkód (Canny)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges_thick = cv2.dilate(edges, kernel, iterations=1)

    edges_colored = np.zeros_like(frame)
    edges_colored[:, :, 2] = edges_thick  # czerwony
    overlay = cv2.addWeighted(overlay, 1.0, edges_colored, alpha_edges, 0)

    # 2. Wykryte obiekty
    for xyxy, cls_name, conf in boxes:
        x1, y1, x2, y2 = map(int, xyxy)
        box_h = y2 - y1

        # Kolor boxa
        if cls_name == "person":
            prox = np.clip(box_h / h, 0.0, 1.0)
            color_box = (int(255*prox), int(255*prox), int(255*prox))  # Biały→czarny
        else:
            prox = np.clip(box_h / h, 0.0, 1.0)
            color_box = (0, 0, int(180 + 75*prox))  # czerwone odcienie

        # Box i etykieta
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_box, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)

    return overlay

# --------- Główna pętla ---------
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    raise RuntimeError("Nie można otworzyć strumienia")

# Ustawienie rozdzielczości kamery
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

mode = "normal"
print("Naciśnij 't' aby przełączyć tryb (normal/przetworzony), ESC aby wyjść.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detekcja YOLO
    results = model(frame)
    res = results[0]
    boxes = []
    if hasattr(res, "boxes"):
        for box in res.boxes:
            xyxy = box.xyxy.cpu().numpy().flatten()
            conf = float(box.conf.cpu().numpy())
            cls_idx = int(box.cls.cpu().numpy()[0])
            cls_name = model.names.get(cls_idx, str(cls_idx)) if hasattr(model, "names") else str(cls_idx)
            if conf >= MIN_CONF and cls_name in CLASSES_KEEP:
                boxes.append((xyxy, cls_name, conf))

    # Tryby wyświetlania
    display_frame = frame if mode == "normal" else draw_hud(frame, boxes)

    # Wyświetlenie
    cv2.imshow("RC Czołg - HUD", display_frame)

    # Obsługa klawiszy
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('t'):  # przełączanie trybu
        mode = "processed" if mode == "normal" else "normal"
        print(f"Tryb zmieniony na: {mode}")

cap.release()
cv2.destroyAllWindows()
