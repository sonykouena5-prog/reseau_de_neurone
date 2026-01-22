import streamlit as st
import cv2
import numpy as np
import os
import zipfile

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

    DB_DIR = "iris_db"
    IMG_DIR = "iris_images"
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    menu = st.sidebar.radio("Menu", [
        "📸 Enregistrer iris",
        "🔍 Tester iris",
        "📋 Afficher base",
        "🗑️ Supprimer iris",
        "♻️ Mettre à jour iris",
        "⬇️ Télécharger base ZIP",
        "⬆️ Importer base ZIP"
    ])

    # === Fonctions utilitaires ===
    def detect_eyes(image):
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        return [image[y:y+h, x:x+w] for (x,y,w,h) in eyes]

    def segment_iris_circles(eye_image):
        gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=100, param2=15, minRadius=10, maxRadius=80)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                return eye_image[y-r:y+r, x-r:x+r]
        return None

    def segment_iris_threshold(eye_image):
        gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            (x,y,w,h) = cv2.boundingRect(c)
            return eye_image[y:y+h, x:x+w]
        return None

    def gabor_features(iris):
        gray = cv2.cvtColor(iris, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        kernels = []
        for theta in np.arange(0, np.pi, np.pi / 4):
            kern = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kernels.append(kern)
        features = []
        for kern in kernels:
            fimg = cv2.filter2D(gray, cv2.CV_8UC3, kern)
            hist = cv2.calcHist([fimg], [0], None, [32], [0,256])
            features.extend(hist.flatten())
        return np.array(features)

    def compare_codes(code1, code2):
        b1 = (code1 > np.mean(code1)).astype(int)
        b2 = (code2 > np.mean(code2)).astype(int)
        return 1 - np.sum(b1 != b2) / len(b1)

    # === Enregistrer iris ===
    if menu == "📸 Enregistrer iris":
        person_name = st.text_input("Nom de la personne")
        frame = st.camera_input("Capture une image")
        if frame and person_name:
            image = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), 1)
            eyes = detect_eyes(image)
            if eyes:
                iris = segment_iris_circles(eyes[0]) or segment_iris_threshold(eyes[0])
                if iris is not None:
                    st.image(iris, caption="Iris capturé", use_column_width=True)
                    features = gabor_features(iris)
                    np.save(os.path.join(DB_DIR, f"{person_name}.npy"), features)
                    cv2.imwrite(os.path.join(IMG_DIR, f"{person_name}.jpg"), iris)
                    st.success(f"✅ Iris enregistré pour {person_name}")
                else:
                    st.error("❌ Impossible de segmenter l’iris")
            else:
                st.error("❌ Aucun œil détecté")

    # === Tester iris ===
    elif menu == "🔍 Tester iris":
        frame = st.camera_input("Capture une image pour tester")
        if frame:
            image = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), 1)
            eyes = detect_eyes(image)
            if eyes:
                iris = segment_iris_circles(eyes[0]) or segment_iris_threshold(eyes[0])
                if iris is not None:
                    st.image(iris, caption="Iris testé", use_column_width=True)
                    test_features = gabor_features(iris)
                    users = [f for f in os.listdir(DB_DIR) if f.endswith(".npy")]
                    if users:
                        best_score = -1
                        best_user = None
                        for user_file in users:
                            db_features = np.load(os.path.join(DB_DIR, user_file))
                            score = compare_codes(test_features, db_features)
                            if score > best_score:
                                best_score = score
                                best_user = user_file.replace(".npy","")
                        st.write(f"Score de similarité : {best_score:.2f}")
                        if best_score > 0.8:
                            st.success(f"✅ Iris reconnu : {best_user}")
                        else:
                            st.error("❌ Iris non reconnu")
                    else:
                        st.warning("⚠️ Base vide, enregistrez d’abord un iris")
                else:
                    st.error("❌ Impossible de segmenter l’iris")
            else:
                st.error("❌ Aucun œil détecté")

    # === Afficher base ===
    elif menu == "📋 Afficher base":
        users = [f.replace(".npy","") for f in os.listdir(DB_DIR) if f.endswith(".npy")]
        if users:
            for user in users:
                st.write(f"👤 {user}")
                img_path = os.path.join(IMG_DIR, f"{user}.jpg")
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"Iris de {user}", width=150)
        else:
            st.warning("⚠️ Aucun utilisateur enregistré")

    # === Supprimer iris ===
    elif menu == "🗑️ Supprimer iris":
        users = [f.replace(".npy","") for f in os.listdir(DB_DIR) if f.endswith(".npy")]
        user_to_delete = st.selectbox("Choisir un utilisateur à supprimer", users)
        if st.button("Supprimer"):
            os.remove(os.path.join(DB_DIR, f"{user_to_delete}.npy"))
            if os.path.exists(os.path.join(IMG_DIR, f"{user_to_delete}.jpg")):
                os.remove(os.path.join(IMG_DIR, f"{user_to_delete}.jpg"))
            st.success(f"✅ Iris supprimé pour {user_to_delete}")

    # === Mettre à jour iris ===
    elif menu == "♻️ Mettre à jour iris":
        users = [f.replace(".npy","") for f in os.listdir(DB_DIR) if f.endswith(".npy")]
        user_to_update = st.selectbox("Choisir un utilisateur à mettre à jour", users)
        frame = st.camera_input("Nouvelle capture")
        if frame and user_to_update:
            image = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), 1)
            eyes = detect_eyes(image)
            if eyes:
                iris = segment_iris_circles(eyes[0]) or segment_iris_threshold(eyes[0])
                if iris is not None:
                    st.image(iris, caption="Nouvel iris", use_column_width=True)
                    features = gabor_features(iris)
                    np.save(os.path.join(DB_DIR, f"{user_to_update}.npy"), features)
                    cv2.imwrite(os.path.join(IMG_DIR, f"{user_to_update}.jpg"), iris)
                    st.success(f"✅ Iris détecté dans l’œil {i+1}")

                        # === Télécharger base ZIP ===
    elif menu == "⬇️ Télécharger base ZIP":
        zip_filename = "iris_base.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            # Ajouter tous les fichiers de la base
            for folder in [DB_DIR, IMG_DIR]:
                for file in os.listdir(folder):
                    zipf.write(os.path.join(folder, file))
        with open(zip_filename, "rb") as f:
            st.download_button(
                label="⬇️ Télécharger la base complète",
                data=f,
                file_name=zip_filename,
                mime="application/zip"
            )

    # === Importer base ZIP ===
    elif menu == "⬆️ Importer base ZIP":
        uploaded_zip = st.file_uploader("Importer un fichier ZIP", type="zip")
        if uploaded_zip is not None:
            with zipfile.ZipFile(uploaded_zip, 'r') as zipf:
                zipf.extractall(".")  # extrait dans le dossier courant
            st.success("✅ Base importée avec succès")
