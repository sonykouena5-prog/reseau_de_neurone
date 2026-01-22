import streamlit as st
import cv2
import numpy as np
import os
import zipfile
from deepface import DeepFace

# === Authentification simple ===
USERS = {
    "sony": "sony@1342",
    "admin": "admin123"
}

st.sidebar.title("🔑 Connexion")
username = st.sidebar.text_input("Nom d'utilisateur")
password = st.sidebar.text_input("Mot de passe", type="password")
login_button = st.sidebar.button("Se connecter")

if login_button:
    if username in USERS and USERS[username] == password:
        st.session_state["authenticated"] = True
        st.session_state["user"] = username
        st.success(f"Bienvenue {username} 👋")
    else:
        st.error("❌ Identifiants incorrects")

if "authenticated" in st.session_state and st.session_state["authenticated"]:
    st.sidebar.success("Connecté ✅")

    DB_DIR = "faces_db"
    os.makedirs(DB_DIR, exist_ok=True)

    MODEL_NAME = "Facenet512"
    MODEL = DeepFace.build_model(MODEL_NAME)

    menu = st.sidebar.radio("Menu", [
        "📸 Enregistrer visage",
        "🔍 Tester reconnaissance",
        "🗑️ Supprimer visage",
        "♻️ Mettre à jour visage",
        "📋 Afficher base",
        "⬇️ Télécharger base ZIP",
        "⬆️ Importer base ZIP",
        "👁️ Détecter yeux",
        "🌐 Segmenter iris"
    ])

    threshold = st.sidebar.slider("Tolérance de reconnaissance", 0.2, 0.6, 0.4)

    def save_image(image, name, index=None):
        image = cv2.resize(image, (300, 300))
        filename = f"{name}_{index}.jpg" if index else f"{name}.jpg"
        path = os.path.join(DB_DIR, filename)
        cv2.imwrite(path, image)
        return path

    def detect_eyes(image):
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        eye_regions = []
        for (x, y, w, h) in eyes:
            eye = image[y:y+h, x:x+w]
            eye_regions.append(eye)
        return eye_regions

    def segment_iris(eye_image):
        gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=20, maxRadius=40)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                iris = eye_image[y-r:y+r, x-r:x+r]
                return iris
        return None

    # === Enregistrer visage ===
    if menu == "📸 Enregistrer visage":
        person_name = st.text_input("Nom de la personne")
        nb_images = st.number_input("Nombre d'images à capturer", min_value=1, max_value=5, value=1)
        for i in range(nb_images):
            frame = st.camera_input(f"Capture image {i+1}")
            if frame and person_name:
                image = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), 1)
                save_image(image, person_name, i+1)
                st.success(f"✅ Image {i+1} enregistrée pour {person_name}")
                st.image(image, caption=f"{person_name} - Image {i+1}", use_column_width=True)

    # === Tester reconnaissance ===
    elif menu == "🔍 Tester reconnaissance":
        frame = st.camera_input("Capture une image pour tester")
        if frame:
            image = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), 1)
            image = cv2.resize(image, (300, 300))
            test_path = "test.jpg"
            cv2.imwrite(test_path, image)

            try:
                results = DeepFace.find(
                    img_path=test_path,
                    db_path=DB_DIR,
                    model_name=MODEL_NAME,
                    model=MODEL,
                    distance_metric="cosine",
                    enforce_detection=False
                )

                if results and len(results) > 0 and len(results[0]) > 0:
                    best_match = results[0].iloc[0]
                    if best_match["distance"] < threshold:
                        user_name = os.path.basename(best_match["identity"]).split("_")[0].replace(".jpg","")
                        st.success(f"✅ Visage reconnu : {user_name}")
                        st.image(image, caption="Visage reconnu ✅", use_column_width=True)
                    else:
                        st.error("❌ Visage non reconnu (distance trop élevée)")
                else:
                    st.error("❌ Aucun visage correspondant trouvé")

            except Exception as e:
                st.warning(f"Erreur lors de la recherche : {str(e)}")

    # === Supprimer visage ===
    elif menu == "🗑️ Supprimer visage":
        users = [f for f in os.listdir(DB_DIR) if f.endswith(".jpg")]
        user_to_delete = st.selectbox("Choisir un fichier à supprimer", users)
        if st.button("Supprimer"):
            os.remove(os.path.join(DB_DIR, user_to_delete))
            st.success(f"✅ {user_to_delete} supprimé")

    # === Mettre à jour visage ===
    elif menu == "♻️ Mettre à jour visage":
        users = [f for f in os.listdir(DB_DIR) if f.endswith(".jpg")]
        user_to_update = st.selectbox("Choisir un fichier à mettre à jour", users)
        frame = st.camera_input("Nouvelle image")
        if frame and user_to_update:
            image = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), 1)
            save_image(image, user_to_update.replace(".jpg",""))
            st.success(f"✅ Visage mis à jour pour {user_to_update}")
            st.image(image, caption=f"Nouveau visage : {user_to_update}", use_column_width=True)

    # === Afficher base ===
    elif menu == "📋 Afficher base":
        users = [f for f in os.listdir(DB_DIR) if f.endswith(".jpg")]
        st.subheader("Utilisateurs enregistrés")
        st.write(users if users else "Aucun utilisateur enregistré")

    # === Export ZIP ===
    elif menu == "⬇️ Télécharger base ZIP":
        zip_path = "faces_db.zip"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for file in os.listdir(DB_DIR):
                zipf.write(os.path.join(DB_DIR, file), file)
        with open(zip_path, "rb") as f:
            st.download_button(
                label="⬇️ Télécharger la base ZIP",
                data=f,
                file_name="faces_db.zip",
                mime="application/zip"
            )
        st.success("✅ Base compressée et prête au téléchargement")

    # === Import ZIP ===
    elif menu == "⬆️ Importer base ZIP":
        uploaded_zip = st.file_uploader("Importer un fichier ZIP", type="zip")
        if uploaded_zip is not None:
            zip_path = "import_faces.zip"
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            with zipfile.ZipFile(zip_path, "r") as zipf:
                zipf.extractall(DB_DIR)
            st.success("✅ Base restaurée avec succès")
            users = [f for f in os.listdir(DB_DIR) if f.endswith(".jpg")]
            st.write("Utilisateurs importés :", users if users else "Aucun")

    # === Détecter yeux ===
    elif menu == "👁️ Détecter yeux":
        frame = st.camera_input("Capture une image pour détecter les yeux")
        if frame:
            image = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), 1)
            eyes = detect_eyes(image)
            if eyes:
                st.success(f"✅ {len(eyes)} yeux détectés")
                for i, eye in enumerate(eyes):
                    st.image(eye, caption=f"Oeil {i+1}", use_column_width=True)
            else:
                st.error("❌ Aucun oeil détecté")

    # === Segmenter iris ===
    elif menu == "🌐 Segmenter iris":
        frame = st.camera_input("Capture une image pour segmenter l’iris")
        if frame:
            image = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), 1)
            eyes = detect_eyes(image)
            if eyes:
                for i, eye in enumerate(eyes):
                    iris = segment_iris(eye)
                    if iris is not None and iris.size > 0:
                        st.success(f"✅ Iris détecté dans l’œil {i+1}")
