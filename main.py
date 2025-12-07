import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk
import datetime
from pathlib import Path
import imageio as iio 
import threading # <-- DODANY IMPORT

# --------- USTAWIENIA ---------
STREAM_URL = 0
MODEL_WEIGHTS = "yolov11n.pt"
CLASSES_KEEP = ["person"]
MIN_CONF = 0.70

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
RECORD_FPS = 7 # Stała, docelowa prędkość zapisu FPS

ALPHA_EDGES = 0.5
KERNEL_SIZE = 2

# --------- Przygotowanie modelu ---------
try:
    model = YOLO(MODEL_WEIGHTS)
except Exception as e:
    print(f"Błąd ładowania modelu YOLO: {e}")
    class DummyModel:
        def __call__(self, frame, verbose=False):
            class DummyResult:
                boxes = type('DummyBoxes', (object,), {'xyxy': np.array([]), 'conf': np.array([]), 'cls': np.array([])})()
            return [DummyResult()]
        names = {}
    model = DummyModel()


# --------- HUD ---------
def draw_hud(frame, boxes, alpha_edges=ALPHA_EDGES, kernel_size=KERNEL_SIZE):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges_thick = cv2.dilate(edges, kernel, 1)

    edges_colored = np.zeros_like(frame)
    edges_colored[:, :, 2] = edges_thick 
    overlay = cv2.addWeighted(overlay, 1.0, edges_colored, alpha_edges, 0)

    for xyxy, cls_name, conf in boxes:
        x1, y1, x2, y2 = map(int, xyxy)
        box_h = y2 - y1

        if cls_name == "person":
            prox = np.clip(box_h / h, 0.0, 1.0)
            color_box = (int(255*prox), int(255*prox), int(255*prox))
        else:
            prox = np.clip(box_h / h, 0.0, 1.0)
            color_box = (0, 0, int(180 + 75*prox))

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_box, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(overlay, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)
    return overlay

# --------- Kamera ---------
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    raise RuntimeError("Nie można otworzyć strumienia wideo.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

mode = "normal"
scale = 1.0
recording = False
video_writer = None 

# --------- Tkinter ---------
root = tk.Tk()
root.title("RC Czołg - HUD (Tkinter)")

canvas = tk.Label(root, bg="black", anchor="center")
canvas.pack(expand=True, fill="both")

fullscreen = False

# --------- FUNKCJA ZAMYKANIA WĄTKU ---------
def close_video_writer_in_thread(writer):
    """Funkcja uruchamiana w osobnym wątku do zamykania pliku wideo."""
    print("Zamykanie pliku wideo w tle... Proszę czekać.")
    try:
        writer.close()
    except Exception as e:
        print(f"Błąd podczas zamykania pliku wideo: {e}")
    print("Zamykanie pliku wideo zakończone.")


# --------- Funkcje sterujące ---------
def toggle_mode(event=None):
    global mode
    mode = "processed" if mode == "normal" else "normal"
    print(f"Tryb zmieniony na: {mode}")

def toggle_fullscreen(event=None):
    global fullscreen
    fullscreen = not fullscreen
    root.attributes("-fullscreen", fullscreen)

def quit_app(event=None):
    global video_writer
    cap.release()
    
    # Użycie wątku, jeśli nagrywanie jest aktywne
    if recording and video_writer is not None:
        threading.Thread(target=close_video_writer_in_thread, args=(video_writer,)).start()
        # Nie musimy czekać, po prostu niszczymy GUI
        
    root.destroy()

def show_message(text, duration=2000):
    """Wyświetla komunikat tymczasowy na kilka sekund, Label tworzony dynamicznie."""
    msg = tk.Label(root, text=text, fg="yellow", bg="black", font=("Arial", 16))
    msg.place(relx=0.5, rely=0.05, anchor="n") 
    root.after(duration, msg.destroy) 

def toggle_record(event=None):
    global recording, video_writer
    downloads_path = Path.home() / "Downloads" 
    downloads_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if recording:
        # Zatrzymywanie nagrywania
        if video_writer is not None:
            # Użycie wątku do zamknięcia writer'a, aby nie blokować GUI
            threading.Thread(target=close_video_writer_in_thread, args=(video_writer,)).start()
            
            video_writer = None # Obiekt jest teraz w wątku, resetujemy globalną referencję
        
        recording = False
        print("Zakończono nagrywanie.")
        show_message("Recording stopped")
    else:
        # Rozpoczynanie nagrywania
        file_path = downloads_path / f"record_{timestamp}.mp4"
        
        try:
            video_writer = iio.get_writer(
                str(file_path), 
                mode='I',         
                codec='libx264',
                quality=7,        
                ffmpeg_params=[
                    '-r', str(RECORD_FPS), 
                    '-pix_fmt', 'yuv420p'  
                ]
            )
            recording = True
            print(f"Rozpoczęto nagrywanie: {file_path} @ {RECORD_FPS} FPS")
            show_message("Recording started")
        except Exception as e:
            print(f"BŁĄD: Nie można zainicjować imageio/FFmpeg: {e}")
            show_message("Recording FAILED")

# --------- Zoom i Pan (bez zmian) ---------
def zoom_in(event=None):
    global scale
    if (event.state & 0x4): 
        scale = min(scale + 0.1, 3.0)
        print(f"Zoom: {scale:.1f}x")

def zoom_out(event=None):
    global scale
    if (event.state & 0x4):  
        scale = max(scale - 0.1, 0.1)
        print(f"Zoom: {scale:.1f}x")

last_x = 0
last_y = 0
offset_x = 0
offset_y = 0

def start_pan(event):
    global last_x, last_y
    last_x = event.x
    last_y = event.y

def do_pan(event):
    global offset_x, offset_y, last_x, last_y
    dx = event.x - last_x
    dy = event.y - last_y
    offset_x += dx
    offset_y += dy
    last_x = event.x
    last_y = event.y

# --------- Bindy ---------
root.bind("t", toggle_mode)
root.bind("<F11>", toggle_fullscreen)
root.bind("<f>", toggle_fullscreen)
root.bind("<Escape>", quit_app)
root.bind("r", toggle_record)

root.protocol("WM_DELETE_WINDOW", quit_app) 

root.bind("<plus>", zoom_in)
root.bind("<KP_Add>", zoom_in)
root.bind("<minus>", zoom_out)
root.bind("<KP_Subtract>", zoom_out)

canvas.bind("<ButtonPress-1>", start_pan)
canvas.bind("<B1-Motion>", do_pan)

# --------- Aktualizacja klatek ---------
imgtk = None
start_time = datetime.datetime.now()
frame_count = 0
current_fps = 0.0

def update_frame():
    global scale, video_writer, canvas, offset_x, offset_y, imgtk, start_time, frame_count, current_fps

    ret, frame = cap.read()
    if not ret:
        root.after(500, update_frame)
        return

    # Obliczanie i aktualizowanie RZECZYWISTEGO FPS
    frame_count += 1
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    if elapsed_time >= 1.0:
        current_fps = frame_count / elapsed_time
        root.title(f"RC Czołg - HUD | Actual FPS: {current_fps:.1f}")
        start_time = datetime.datetime.now()
        frame_count = 0

    # Krok 1: Wstępne przetwarzanie YOLO
    results = model(frame, verbose=False)
    res = results[0]

    boxes = []
    if hasattr(res, "boxes") and len(res.boxes) > 0:
        for box in res.boxes:
            xyxy = box.xyxy.cpu().numpy().flatten()
            
            if not isinstance(model.names, dict):
                 cls_name = "unknown"
            else:
                 cls_idx = int(box.cls.cpu().numpy().item())
                 cls_name = model.names.get(cls_idx, str(cls_idx))
            
            conf = box.conf.cpu().numpy().item() 

            if conf >= MIN_CONF and cls_name in CLASSES_KEEP:
                boxes.append((xyxy, cls_name, conf))

    # Krok 2: Ustalenie ramki do wyświetlania/nagrywania
    if mode == "processed":
        display_frame = draw_hud(frame.copy(), boxes)
    else:
        display_frame = frame.copy() 

    # Krok 3: NANIESIENIE TEKSTU FPS (na klatkę do wyświetlenia/nagrywania)
    if current_fps > 0:
        fps_label = f"FPS: {current_fps:.1f}"
        cv2.putText(display_frame, fps_label, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        if recording:
             cv2.putText(display_frame, "REC", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


    # Krok 4: Nagrywanie (używamy display_frame, konwertujemy BGR -> RGB)
    if recording and video_writer is not None:
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        video_writer.append_data(frame_rgb)

    # Krok 5: Skalowanie i wyświetlanie (używamy display_frame)
    h, w = display_frame.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    display_frame_resized = cv2.resize(display_frame, (new_w, new_h))
    rgb = cv2.cvtColor(display_frame_resized, cv2.COLOR_BGR2RGB)
    
    # Konwersja do ImageTk
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.imgtk = imgtk
    canvas.config(image=imgtk)

    root.after(10, update_frame)

# --------- Start ---------
print("--- Sterowanie ---")
print("F11 / f - Pełny ekran")
print("t - Przełącz tryb (Normalny / HUD/linie/ramki)")
print("r - Rozpocznij/Zakończ nagrywanie (Nagrywa to, co widzisz na ekranie)")
print("Ctrl + +/- - Zoom natychmiastowy")
print("ESC / X (przycisk) - Wyjście z aplikacji")
print("------------------")
update_frame()
root.mainloop()