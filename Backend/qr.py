import cv2
import time
import re
import pickle
import pandas as pd
from urllib.parse import urlparse

# === Load trained model ===
model = pickle.load(open("XGBoostClassifier.pickle.dat", "rb"))

# === URL Feature Extraction Function ===
def extract_features(url):
    """Extracts features from a single URL — must match the features used during training."""
    features = {}

    # 1. Have_IP: Check if the URL contains an IP address
    ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'
    features['Have_IP'] = 1 if re.search(ip_pattern, url) else 0

    # 2. Have_At: Check if '@' symbol is present
    features['Have_At'] = 1 if '@' in url else 0

    # 3. URL length
    features['URL_Length'] = len(url)

    # 4. URL depth (count of '/')
    features['URL_Depth'] = url.count('/')

    # 5. Presence of redirection ('//' after protocol)
    features['Redirection'] = 1 if '//' in url[7:] else 0

    # 6. HTTPS check
    features['https_Domain'] = 1 if url.startswith("https") else 0

    # 7. TinyURL or shortener services
    shortening_services = r"bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co|is\.gd|buff\.ly|adf\.ly"
    features['TinyURL'] = 1 if re.search(shortening_services, url) else 0

    # 8. Prefix/Suffix in domain
    features['Prefix/Suffix'] = 1 if '-' in urlparse(url).netloc else 0

    # 9–16: Placeholder features
    features['DNS_Record'] = 1
    features['Web_Traffic'] = 1
    features['Domain_Age'] = 1
    features['Domain_End'] = 1 if url.endswith(('.com', '.org', '.net')) else 0
    features['iFrame'] = 0
    features['Mouse_Over'] = 0
    features['Right_Click'] = 0
    features['Web_Forwards'] = 0

    return pd.DataFrame([features])

# === Prediction Function ===
def predict_label(url):
    """Predict if the given URL is phishing or legitimate."""
    features = extract_features(url)
    prediction = model.predict(features)[0]
    return "PHISHING" if prediction == 1 else "LEGITIMATE"

# === QR Camera Function ===
def read_and_classify():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera (index 0). Try changing index to 1 or another number.")
        return

    detector = cv2.QRCodeDetector()
    last = {"data": None, "label": None, "time": 0}

    print("📷 Point the camera at a QR code. Press 'q' to quit.\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        data, bbox, _ = detector.detectAndDecode(frame)
        if bbox is not None and len(bbox) > 0:
            pts = bbox.astype(int).reshape(-1, 2)
            for i in range(len(pts)):
                cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 2)

        if data:
            if data != last["data"] or (time.time() - last["time"] > 5):
                last["data"] = data
                last["time"] = time.time()
                try:
                    label = predict_label(data)
                except Exception as e:
                    label = f"ERROR: {e}"
                last["label"] = label
                print(f"Detected QR: {data}\n→ {label}\n")

            text = f"{last['label']}: {data}"
            if len(text) > 80:
                text = text[:77] + "..."
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("QR Scanner + Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    read_and_classify()

