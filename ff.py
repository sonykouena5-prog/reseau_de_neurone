import streamlit as st
import numpy as np
from PIL import Image
import os

def normalize_boxes(boxes, img_size, relative=False):
    w, h = img_size
    normalized = []
    for (x, y, w_box, h_box) in boxes:
        if relative:
            normalized.append((int(x * w), int(y * h), int(w_box * w), int(h_box * h)))
        else:
            normalized.append((x, y, w_box, h_box))
    return normalized

def detect_with_solutions(img):
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        img_np = np.array(img)
        results = detector.process(img_np)
        if not results.detections:
            return []
        boxes = []
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            boxes.append((bbox.xmin, bbox.ymin, bbox.width, bbox.height))
        return normalize_boxes(boxes, img.size, relative=True)

def ensure_model(model_path="face_detection_short_range.tflite"):
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/face_detection/face_detection_short_range/float16/1/face_detection_short_range.tflite"
        st.info("Téléchargement du modèle MediaPipe...")
        import urllib.request
        urllib.request.urlretrieve(url, model_path)
    return model_path

def detect_with_tasks(img):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = ensure_model()
    base_options = vision.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)
    detector = vision.FaceDetector.create_from_options(options)

    img_np = np.array(img)
    mp_image = mp_python.Image(image_format=mp_python.ImageFormat.SRGB, data=img_np)
    detection_result = detector.detect(mp_image)

    boxes = []
    for face in detection_result.detections:
        bbox = face.bounding_box
        boxes.append((bbox.origin_x, bbox.origin_y, bbox.width, bbox.height))
    return normalize_boxes(boxes, img.size, relative=False)

def detect_auto(img):
    try:
        import mediapipe as mp
    except Exception as e:
        st.error("Mediapipe introuvable. Installe-le avec : `pip install mediapipe`")
        return []

    if hasattr(mp, "solutions"):
        try:
            st.caption("Utilisation de : mediapipe.solutions (ancienne API)")
            return detect_with_solutions(img)
        except Exception as e:
            st.warning(f"Échec avec mediapipe.solutions : {e}")

    try:
        st.caption("Utilisation de : mediapipe.tasks (nouvelle API)")
        return detect_with_tasks(img)
    except Exception as e:
        st.warning(f"Échec avec mediapipe.tasks : {e}")
        return []

# Interface Streamlit
st.title("Détection de visage avec MediaPipe")
uploaded_file = st.file_uploader("Choisis une image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Image chargée", use_column_width=True)

    boxes = detect_auto(img)
    if boxes:
        st.success(f"{len(boxes)} visage(s) détecté(s)")
        st.write("Coordonnées (x, y, largeur, hauteur) :", boxes)
    else:
        st.warning("Aucun visage détecté.")
