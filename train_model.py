import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def kantenerkennung(face):   # Funktion für Bildverarbeitung
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) # Gesicht wird in Graustufe umgewandelt
    gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=10) # Kontrast wird erhöht damit Kanten besser erkannt werden
    blur = cv2.GaussianBlur(gray, (3, 3), 0) # Bild wird geglättet damit keine falschen Kanten erkannt werden
    edges = cv2.Canny(blur, 50, 150)# Kantendetektion - starke Helligkeitswechsel werden gesucht.

    return gray, blur, edges

def extract_features(face):
    face = cv2.resize(face, (100, 100)) # Gesichter werden auf die selbe Größe gebracht

    gray, blur, edges = kantenerkennung(face)

    # ganzes Gesicht als Kantenbild
    face_edges_small = cv2.resize(edges, (50, 50)) #Gesicht Bild wird kleiner gemacht für weniger Daten

    # Mundbereich zusätzlich
    mouth = edges[50:95, 15:85]
    mouth_small = cv2.resize(mouth, (40, 25)) # Mundbereiche wird auch kleiner gemacht

    face_features = face_edges_small.flatten() / 255.0# /255 werden dann Werte zwischen 0 und 1
    mouth_features = mouth_small.flatten() / 255.0# Aus einem Bild wird eine lange Zahlenliste gemacht

    features = np.concatenate([face_features, mouth_features])# Merkmale von Gesicht ung Mund werden Zusammengefügt

    return features

X = []
y = []

labels = ["happy", "sad", "neutral"]

for label in labels:
    folder = f"dataset/{label}"

    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)

            if img is not None:
                features = extract_features(img)
                X.append(features)
                y.append(label)

X = np.array(X)
y = np.array(y)

print("Anzahl Trainingsbilder:", len(X))

if len(X) < 10:
    print("Zu wenige Bilder. Sammle zuerst mehr Trainingsbilder.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3) #3 ähnliche Bilder es wird geschaut welche Ergebniss zutrifft
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Genauigkeit:", round(accuracy * 100, 2), "%")

joblib.dump(model, "mimik_model.pkl") # Das speichert trainiertes Model in dieser Datei
print("Modell gespeichert als mimik_model.pkl")