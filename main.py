import cv2
import numpy as np
import joblib
import os

from collections import deque, Counter
import time


def kantenerkennung(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=10)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    return gray, blur, edges


def extract_features(face):
    # Gesicht immer auf gleiche Größe bringen
    face = cv2.resize(face, (100, 100))

    # Kantenerkennung auf Gesicht anwenden
    gray, blur, edges = kantenerkennung(face)

    # Ganzes Gesicht als Kantenbild verwenden
    face_edges_small = cv2.resize(edges, (50, 50))

    # Zusätzlich Mundbereich ausschneiden
    mouth = edges[50:95, 15:85]
    mouth_small = cv2.resize(mouth, (40, 25))

    # Alles in Zahlen umwandeln
    face_features = face_edges_small.flatten() / 255.0
    mouth_features = mouth_small.flatten() / 255.0

    # Beide Merkmale zusammenfügen
    features = np.concatenate([face_features, mouth_features])

    return features, gray, blur, edges, mouth

# Modell prüfen
if not os.path.exists("mimik_model.pkl"):
    print("FEHLER: mimik_model.pkl fehlt.")
    print("Starte zuerst train_model.py, damit das Modell erstellt wird.")
    exit()

model = joblib.load("mimik_model.pkl") #Hier wird das trainierte Modell geladen
print("Modell geladen.")


# Gesichtserkennung
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("FEHLER: Gesichtserkennung konnte nicht geladen werden.")
    exit()


# Webcam wird gestarted
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Webcam konnte nicht geöffnet werden.")
    exit()

print("Programm läuft. Drücke q zum Beenden.")

prediction_history = deque()
stable_prediction = "Warte..."
last_update_time = time.time()

while True:
    ret, frame = cap.read() #aktuelles Webcam-Bild wird gelesen.

    if not ret or frame is None:
        print("Kein Bild von der Webcam.")
        break

    frame = cv2.flip(frame, 1)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = (face_cascade.detectMultiScale #Gesicht wird wieder erkannt
    (
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(80, 80)
    ))

    status_text = "Kein Gesicht erkannt"

    if len(faces) > 0:
        # größtes Gesicht nehmen
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0] # Wenn Gesicht erkennt wurde bekommst du Werte

        face = frame[y:y+h, x:x+w] #Gesicht wird ausgeschnitten

        features, gray, blur, edges, mouth = extract_features(face)#?

        prediction = model.predict([features])[0] # Mimik erkennung

        # aktuelle Vorhersage mit Zeit speichern
        current_time = time.time()
        prediction_history.append((current_time, prediction))

        # nur Vorhersagen der letzten 1 Sekunde behalten
        while prediction_history and current_time - prediction_history[0][0] > 0.5:
            prediction_history.popleft()

        if current_time - last_update_time >= 0.5:
            predictions_only = [p for t, p in prediction_history]

            if len(predictions_only) > 0:
                most_common_label, count = Counter(predictions_only).most_common(1)[0]

                # Nur wechseln, wenn mindestens 60 % gleich sind
                if count / len(predictions_only) >= 0.6:
                    stable_prediction = most_common_label

            last_update_time = current_time

        status_text = f"Mimik: {stable_prediction}" #Mimik wird als Text ausgegeben

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #Grüne Rechteck um Gesicht

        cv2.putText(
            frame,
            status_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow("Gesicht - Kantenbild", edges)
        cv2.imshow("Mundbereich - Kanten", mouth)

    cv2.putText(
        frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("Mimik-Erkennung", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()