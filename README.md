# KI-gestützte Mimikerkennung mit Kantendetektion

## Projektbeschreibung

Dieses Projekt erkennt mithilfe einer Webcam einfache Mimiken wie happy, neutral und sad.

Dazu wird zuerst mit OpenCV ein Gesicht im Kamerabild erkannt. Anschließend wird das Gesicht ausgeschnitten und mit klassischer Bildverarbeitung vorbereitet. Das Bild wird in Graustufen umgewandelt, leicht geglättet und danach mit dem Canny-Algorithmus in ein Kantenbild umgerechnet.

Aus diesem Kantenbild werden Merkmale erzeugt. Diese Merkmale werden anschließend von einem KNN-Modell (K-Nearest Neighbors) klassifiziert. Das Modell entscheidet also, ob die aktuelle Mimik am ehesten happy, neutral oder sad ist.

## Verwendete Technologien

- Python
- OpenCV
- NumPy
- scikit-learn
- joblib
- Canny-Kantendetektion
- KNN-Modell zur Klassifizierung

## Dateien

- `collect_data.py`  
  Dient zum Sammeln eigener Trainingsbilder über die Webcam.

- `train_model.py`  
  Trainiert das KNN-Modell mit den gespeicherten Bildern.

- `main.py`  
  Startet die Live-Erkennung der Mimik über die Webcam.

- `mimik_model.pkl`  
  Enthält das bereits trainierte Modell.

- `requirements.txt`  
  Enthält die benötigten Python-Bibliotheken.

## Installation

Zuerst müssen die benötigten Bibliotheken installiert werden:

```bash
pip install -r requirements.txt