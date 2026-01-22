import tkinter as tk
from tkinter import filedialog, messagebox
import mediapipe as mp
import numpy as np
from PIL import Image
import os

DATASET_DIR = "faces_dataset"
mp_face_detection = mp.solutions.face_detection

def detect_face_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_np)
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            return np.array([bbox.xmin, bbox.ymin, bbox.width, bbox.height])
        else:
            return None

def enregistrer_visage():
    file_path = filedialog.askopenfilename(title="Choisir une image")
    if not file_path:
        return
    nom = name_entry.get()
    if not nom:
        messagebox.showwarning("Attention", "Veuillez entrer un nom pour ce visage.")
        return

    embedding = detect_face_embedding(file_path)
    if embedding is not None:
        if not os.path.exists(DATASET_DIR):
            os.makedirs(DATASET_DIR)
        np.save(os.path.join(DATASET_DIR, f"{nom}.npy"), embedding)
        messagebox.showinfo("Succès", f"Visage '{nom}' enregistré avec succès.")
    else:
        messagebox.showerror("Erreur", "Aucun visage détecté dans l'image.")

def tester_reconnaissance():
    file_path = filedialog.askopenfilename(title="Choisir une image à tester")
    if not file_path:
        return

    test_embedding = detect_face_embedding(file_path)
    if test_embedding is None:
        messagebox.showerror("Erreur", "Aucun visage détecté dans l'image de test.")
        return

    for filename in os.listdir(DATASET_DIR):
        if filename.endswith(".npy"):
            known_embedding = np.load(os.path.join(DATASET_DIR, filename))
            distance = np.linalg.norm(test_embedding - known_embedding)
            if distance < 0.1:  # seuil simple
                messagebox.showinfo("Résultat", f"Visage reconnu : {filename[:-4]}")
                return

    messagebox.showinfo("Résultat", "Visage inconnu.")

# Interface Tkinter
root = tk.Tk()
root.title("Application de Reconnaissance Faciale")

tk.Label(root, text="Nom du visage à enregistrer :").pack(pady=5)
name_entry = tk.Entry(root)
name_entry.pack(pady=5)

tk.Button(root, text="📸 Enregistrer un visage", command=enregistrer_visage).pack(pady=10)
tk.Button(root, text="🔍 Tester une image", command=tester_reconnaissance).pack(pady=10)

root.mainloop()
