import streamlit as st
import cv2
import numpy as np
import os
import zipfile

# === Authentification simple ===
USERS = {"sony": "sony@1342", "admin": "admin123"}

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
    def detect_face_and_eyes(image):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes_regions = []
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                eyes_regions.append((ex, roi_color[ey:ey+eh, ex:ex+ew]))
        eyes_regions = sorted(eyes_regions, key=lambda e: e[0])
        return [e[1] for e in eyes_regions]

    def segment_iris(eye_image):
        gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=100, param2=15, minRadius=20, maxRadius=80)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                iris = eye_image[y-r:y+r, x-r:x+r]
                return cv2.resize(iris, (128, 128))
        return cv2.resize(eye_image, (128,128))

    def extract_features(iris):
        gray = cv2.cvtColor(iris, cv2.COLOR_BGR2GRAY)
        features = []
        for theta in np.arange(0, np.pi, np.pi/4):
            kern = cv2.getGaborKernel((21,21), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            fimg = cv2.filter2D(gray, cv2.CV_8UC3, kern)
            hist = cv2.calcHist([fimg], [0], None, [32], [0,256])
            features.extend(hist.flatten())
        return np.array(features)

    def compare_codes(code1, code2, threshold=0.85):
        b1 = (code1 > np.mean(code1)).astype(int)
        b2 = (code2 > np.mean(code2)).astype(int)
        score = 1 - np.sum(b1 != b2) / len(b1)
        return score, score >= threshold

    # === Slider pour seuil ===
    threshold = st.sidebar.slider("Seuil de reconnaissance", 0.7, 0.95, 0.85)

    # === Enregistrement visage ===
    if menu == "📸 Enregistrer iris":
        person_name = st.text_input("Nom de la personne")
        frame = st.camera_input("Capture du visage (cadre réduit)", help="Placez uniquement le visage dans le cadre")
        if frame and person_name:
            image = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), 1)
            eyes = detect_face_and_eyes(image)
            if len(eyes) >= 2:
                for i, eye_label in enumerate(["left","right"]):
                    iris = segment_iris(eyes[i])
                    if iris is not None:
                        np.save(os.path.join(DB_DIR, f"{person_name}_{eye_label}.npy"), extract_features(iris))
                        cv2.imwrite(os.path.join(IMG_DIR, f"{person_name}_{eye_label}.jpg"), iris)
                        st.success(f"✅ Iris {eye_label} enregistré pour {person_name}")
            elif len(eyes) == 1:
                iris = segment_iris(eyes[0])
                if iris is not None:
                    np.save(os.path.join(DB_DIR, f"{person_name}_single.npy"), extract_features(iris))
                    cv2.imwrite(os.path.join(IMG_DIR, f"{person_name}_single.jpg"), iris)
                    st.success(f"✅ Iris unique enregistré pour {person_name}")
            else:
                st.error("❌ Aucun œil détecté")

    # === Test visage ===
    elif menu == "🔍 Tester iris":
        frame = st.camera_input("Capture du visage (cadre réduit)", help="Placez uniquement le visage dans le cadre")
        if frame:
            image = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), 1)
            eyes = detect_face_and_eyes(image)
            if len(eyes) >= 2:
                results = []
                for i, eye_label in enumerate(["left","right"]):
                    iris = segment_iris(eyes[i])
                    if iris is not None:
                        test_features = extract_features(iris)
                        candidates = [f for f in os.listdir(DB_DIR) if f.endswith(f"_{eye_label}.npy")]
                        best_score = -1
                        best_user = None
                        for file in candidates:
                            db_features = np.load(os.path.join(DB_DIR, file))
                            score, match = compare_codes(test_features, db_features, threshold)
                            if score > best_score:
                                best_score = score
                                best_user = file.replace(f"_{eye_label}.npy","")
                        if best_score > threshold:
                            results.append(best_user)
                            st.success(f"✅ Œil {eye_label} reconnu : {best_user} (score {best_score:.2f})")
                            img_path = os.path.join(IMG_DIR, f"{best_user}_{eye_label}.jpg")
                            if os.path.exists(img_path):
                                st.image(img_path, caption=f"Iris {eye_label} enregistré de {best_user}", width=120)
                        else:
                            st.error(f"❌ Œil {eye_label} non reconnu")
                if len(set(results)) == 1 and results:
                    st.success(f"🎯 Reconnaissance confirmée : {results[0]}")
            elif len(eyes) == 1:
                iris = segment_iris(eyes[0])
                if iris is not None:
                    test_features = extract_features(iris)
                    candidates = [f for f in os.listdir(DB_DIR) if f.endswith("_single.npy")]
                    best_score = -1
                    best_user = None
                    for file in candidates:
                        db_features = np.load(os.path.join(DB_DIR, file))
                        score, match = compare_codes(test_features, db_features, threshold)
                        if score > best_score:
                            best_score = score
                            best_user = file.replace("_single.npy","")
                    if best_score > threshold:
                        st.success(f"✅ Iris unique reconnu : {best_user} (score {best_score:.2f})")
                        img_path = os.path.join(IMG_DIR, f"{best_user}_single.jpg")
                        if os.path.exists(img_path):
                            st.image(img_path, caption=f"Iris unique enregistré de {best_user}", width=120)
                    else:
                        st.error("❌ Iris unique non reconnu")
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
                    st.image(img_path, caption=f"Iris de {user}", width=120)
        else:
            st.warning("⚠️ Aucun utilisateur enregistré")

    # === Supprimer iris ===
    elif menu == "🗑️ Supprimer iris":
        users = [f.replace(".npy","") for f in os.listdir(DB_DIR) if f.endswith(".npy")]
        user_to_delete = st.selectbox("Choisir un iris à supprimer", users)
        if st.button("Supprimer"):
            npy_path = os.path.join(DB_DIR, f"{user_to_delete}.npy")
            img_path = os.path.join(IMG_DIR, f"{user_to_delete}.jpg")
            if os.path.exists(npy_path):
                os.remove(npy_path)
            if os.path.exists(img_path):
                os.remove(img_path)
            st.success(f"✅ Iris supprimé : {user_to_delete}")

    # === Mettre à jour iris ===
    elif menu == "♻️ Mettre à jour iris":
        users = [f.replace(".npy","") for f in os.listdir(DB_DIR) if f.endswith(".npy")]
        user_to_update = st.selectbox("Choisir un iris à mettre à jour", users)
        frame = st.camera_input("Nouvelle capture du visage (cadre réduit)")
        if frame and user_to_update:
            image = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), 1)
            eyes = detect_face_and_eyes(image)
            if len(eyes) >= 2:
                iris_left = segment_iris(eyes[0])
                iris_right = segment_iris(eyes[1])
                if iris_left is not None and user_to_update.endswith("left"):
                    np.save(os.path.join(DB_DIR, f"{user_to_update}.npy"), extract_features(iris_left))
                    cv2.imwrite(os.path.join(IMG_DIR, f"{user_to_update}.jpg"), iris_left)
                    st.success(f"✅ Iris gauche mis à jour : {user_to_update}")
                if iris_right is not None and user_to_update.endswith("right"):
                    np.save(os.path.join(DB_DIR, f"{user_to_update}.npy"), extract_features(iris_right))
                    cv2.imwrite(os.path.join(IMG_DIR, f"{user_to_update}.jpg"), iris_right)
                    st.success(f"✅ Iris droit mis à jour : {user_to_update}")
            elif len(eyes) == 1 and user_to_update.endswith("single"):
                iris = segment_iris(eyes[0])
                if iris is not None:
                    np.save(os.path.join(DB_DIR, f"{user_to_update}.npy"), extract_features(iris))
                    cv2.imwrite(os.path.join(IMG_DIR, f"{user_to_update}.jpg"), iris)
                    st.success(f"✅ Iris unique mis à jour : {user_to_update}")
            else:
                st.error("❌ Impossible de détecter l'œil correspondant")

    # === Télécharger base ZIP ===
    elif menu == "⬇️ Télécharger base ZIP":
        zip_filename = "iris_base.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
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
                zipf.extractall(".")
            st.success("✅ Base importée avec succès")
