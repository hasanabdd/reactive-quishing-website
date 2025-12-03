import re
import pickle
import pandas as pd
from urllib.parse import urlparse

# Load your trained model
model = pickle.load(open("XGBoostClassifier.pickle.dat", "rb"))

def extract_features(url):
    features = {}

    ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'
    features['Have_IP'] = 1 if re.search(ip_pattern, url) else 0

    features['Have_At'] = 1 if '@' in url else 0

    features['URL_Length'] = len(url)

    features['URL_Depth'] = url.count('/')

    features['Redirection'] = 1 if '//' in url[7:] else 0

    features['https_Domain'] = 1 if url.startswith("https") else 0

    shortening_services = r"bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co|is\.gd|buff\.ly|adf\.ly"
    features['TinyURL'] = 1 if re.search(shortening_services, url) else 0

    features['Prefix/Suffix'] = 1 if '-' in urlparse(url).netloc else 0

    features['DNS_Record'] = 1
    features['Web_Traffic'] = 1
    features['Domain_Age'] = 1
    features['Domain_End'] = 1 if url.endswith(('.com', '.org', '.net')) else 0
    features['iFrame'] = 0
    features['Mouse_Over'] = 0
    features['Right_Click'] = 0
    features['Web_Forwards'] = 0

    return pd.DataFrame([features])

def classify_url(url):
    features = extract_features(url)
    pred = model.predict(features)[0]
    label = "PHISHING" if pred == 1 else "LEGITIMATE"

    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(features)[0][1])

    return label, int(pred), proba
