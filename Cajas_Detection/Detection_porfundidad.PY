


#Libreria de Yolo
from ultralytics import YOLO
import cv2 


#Lectura del modelo 
model = YOLO("best2.pt")

#configurarion del tamaño de la ventana


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

frame_skip = 6  # Procesar cada 4to cuadro
frame_count = 0

while cap.isOpened():

    status, frame = cap.read()
     
    if not status:
        print("ERROR: No se ha recibido el frame")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Redimensionar la imagen para reducir la carga de procesamiento
    resized_frame = cv2.resize(frame, (640, 640))

    # Realizar la detección con el modelo
    results = model.predict(resized_frame, imgsz=640, conf=0.85)

    notes = results[0].plot()


    cv2.imshow("detected boxes", notes)

    if cv2.waitKey(20) == 27 or not cv2.getWindowProperty("detected boxes", cv2.WND_PROP_VISIBLE) : # & 0xFF == ord("q")
        break

cap.release()
cv2.destroyAllWindows()
    



    
         

    

