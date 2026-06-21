import cv2
import os

# Ordner erstellen, falls sie noch nicht existieren
os.makedirs("dataset/happy", exist_ok=True)
os.makedirs("dataset/sad", exist_ok=True)
os.makedirs("dataset/neutral", exist_ok=True)

face_cascade = cv2.CascadeClassifier( #Gesichtserkennung von OpenCv - fertiges Modell welches Gesichter in einem Bild findet
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #hier mit wird die Webcamp des Localen PCs/Laptops gestarted.

if not cap.isOpened():
    print("Webcam konnte nicht geöffnet werden.")
    exit()

counter = {
    "happy": len(os.listdir("dataset/happy")),
    "sad": len(os.listdir("dataset/sad")),
    "neutral": len(os.listdir("dataset/neutral"))
}

print("Tasten:")
print("h = happy speichern")
print("s = sad speichern")
print("n = neutral speichern")
print("q = beenden")

while True:
    ret, frame = cap.read() # ret - wurde das Bild erfolgreich gelesen?
    # frame - das eigentliche Kamerabild

    if not ret or frame is None:
        print("Kein Kamerabild.")
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Bild in Graustufe umgewandelt.

    faces = face_cascade.detectMultiScale( # OpenCV sucht nach Gesichter im Bild.
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(80, 80)
    )

    face_crop = None

    if len(faces) > 0:
        # größtes Gesicht nehmen
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        face_crop = frame[y:y+h, x:x+w] # Hier wird das Gesicht aus dem ganzen Kamerabild.

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        "h=happy | s=sad | n=neutral | q=quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Webcam - Daten sammeln", frame)

    key = cv2.waitKey(1) & 0xFF

    if face_crop is not None:
        if key == ord("h"): # Wenn gedrückt wird Bild als happy gespeichert.
            counter["happy"] += 1
            path = f"dataset/happy/happy_{counter['happy']}.jpg"
            cv2.imwrite(path, face_crop)
            print("Gespeichert:", path)

        elif key == ord("s"):
            counter["sad"] += 1
            path = f"dataset/sad/sad_{counter['sad']}.jpg"
            cv2.imwrite(path, face_crop)
            print("Gespeichert:", path)

        elif key == ord("n"):
            counter["neutral"] += 1
            path = f"dataset/neutral/neutral_{counter['neutral']}.jpg"
            cv2.imwrite(path, face_crop)
            print("Gespeichert:", path)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()