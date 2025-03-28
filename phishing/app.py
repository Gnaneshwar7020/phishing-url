import os
import logging
import torch
import torch.nn as nn
import numpy as np
import joblib
import tldextract
from urllib.parse import urlparse
import re
from nltk.corpus import words
from collections import Counter
import nltk
from flask import Flask, request, render_template, jsonify, send_file
from flask_cors import CORS
import csv
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)

# Download NLTK words if not already downloaded
nltk.download('words', quiet=True)

# Define brand names and suspicious TLDs
BRAND_NAMES = set([
    "google", "facebook", "amazon", "apple", "microsoft", "twitter", "instagram",
    "linkedin", "paypal", "dropbox", "alibaba", "ebay", "netflix", "youtube",
    "wordpress", "adobe", "cisco", "oracle", "salesforce", "shopify", "walmart",
    "target", "bestbuy", "costco", "tesla", "uber", "airbnb", "booking", "expedia",
    "spotify", "slack", "zoom", "tiktok", "snapchat", "reddit", "pinterest",
    "whatsapp", "telegram", "discord", "skype", "samsung", "sony", "nike", "adidas"
])
SUSPICIOUS_TLDS = set([
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "online", "site",
    "info", "biz", "cc", "pw", "click", "link", "win", "faith", "party",
    "date", "loan", "stream", "download", "review", "bid", "trade"
])

# URL Validation Function
def is_valid_url(url):
    """
    Validates whether the given URL is properly structured.
    """
    try:
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc])
    except Exception:
        return False

# Feature Extraction Function
def extract_features(url):
    features = {}
    parsed_url = urlparse(url)
    ext = tldextract.extract(url)
    netloc = parsed_url.netloc

    # f1–2: Full URL length and hostname length
    features['f1'] = len(url)
    features['f2'] = len(netloc)

    # f3: Presence of IP address in hostname
    features['f3'] = 1 if re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", netloc) else 0

    # f4–20: Count of special characters
    special_chars = ['.', '-', '@', '?', '&', '|', '=', '_', '~', '%', '/', '*', ':', ',', ';', '$', '%20']
    for i, char in enumerate(special_chars, start=4):
        features[f'f{i}'] = url.count(char)

    # f21–24: Count of common terms
    common_terms = ['www', '.com', 'http', '//']
    for i, term in enumerate(common_terms, start=21):
        features[f'f{i}'] = url.count(term)

    # f25: HTTPS token presence
    features['f25'] = 1 if parsed_url.scheme == 'https' else 0

    # f26–27: Ratio of digits in full URL and hostname
    features['f26'] = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0
    features['f27'] = sum(c.isdigit() for c in netloc) / len(netloc) if len(netloc) > 0 else 0

    # f28: Punycode presence
    features['f28'] = 1 if "xn--" in netloc else 0

    # f29: Port presence
    features['f29'] = 1 if parsed_url.port else 0

    # f30–31: TLD position in path and subdomain
    features['f30'] = 1 if ext.suffix in parsed_url.path else 0
    features['f31'] = 1 if ext.suffix in ext.subdomain else 0

    # f32: Abnormal subdomains
    features['f32'] = 1 if re.search(r"w[w]?[0-9]*", ext.subdomain) else 0

    # f33: Number of subdomains
    features['f33'] = len(ext.subdomain.split('.')) if ext.subdomain else 0

    # f34: Prefix/Suffix presence
    features['f34'] = 1 if '-' in ext.domain else 0

    # f35: Random domains (using English words as a reference)
    english_words = set(words.words())
    features['f35'] = 0 if ext.domain.lower() in english_words else 1

    # f36: Shortening service usage
    shorteners = {"bit.ly", "goo.gl", "tinyurl.com"}
    features['f36'] = 1 if any(shortener in url for shortener in shorteners) else 0

    # f37: Path extension presence
    malicious_extensions = {'.txt', '.exe', '.js'}
    features['f37'] = 1 if any(url.endswith(extn) for extn in malicious_extensions) else 0

    # f38–39: Placeholder for redirection counts
    features['f38'] = url.count("redirect")
    features['f39'] = url.count("external")

    # f40–47: NLP features based on tokenization of URL
    tokens = re.findall(r'\b\w+\b', url)
    features['f40'] = len(tokens)
    features['f41'] = sum(Counter(token).most_common(1)[0][1] - 1 for token in tokens if len(token) > 1)
    features['f42'] = min([len(token) for token in tokens], default=0)
    features['f43'] = min([len(part) for part in ext.domain.split('-')], default=0)
    features['f44'] = min([len(part) for part in parsed_url.path.split('/') if part], default=0)
    features['f45'] = max([len(token) for token in tokens], default=0)
    features['f46'] = max([len(part) for part in ext.domain.split('-')], default=0)
    features['f47'] = max([len(part) for part in parsed_url.path.split('/') if part], default=0)

    # f48–50: Average word length in tokens, domain parts, and path parts
    features['f48'] = sum(len(token) for token in tokens) / len(tokens) if tokens else 0
    domain_parts = ext.domain.split('-')
    features['f49'] = sum(len(part) for part in domain_parts) / len(domain_parts) if domain_parts else 0
    path_parts = [part for part in parsed_url.path.split('/') if part]
    features['f50'] = sum(len(part) for part in path_parts) / len(path_parts) if path_parts else 0

    # f51: Phish hints count
    phish_hints = {'wp', 'login', 'includes', 'admin', 'content', 'site', 'images', 'js', 'alibaba', 'css', 'myaccount', 'dropbox', 'themes', 'plugins', 'signin', 'view'}
    features['f51'] = sum(url.count(hint) for hint in phish_hints)

    # f52–54: Brand domains presence in domain, subdomain, and path
    features['f52'] = 1 if any(brand in ext.domain for brand in BRAND_NAMES) else 0
    features['f53'] = 1 if any(brand in ext.subdomain for brand in BRAND_NAMES) else 0
    features['f54'] = 1 if any(brand in parsed_url.path for brand in BRAND_NAMES) else 0

    # f55: Suspicious TLD presence
    features['f55'] = 1 if ext.suffix in SUSPICIOUS_TLDS else 0

    return list(features.values())

# CustomResidualUnit
class CustomResidualUnit(nn.Module):
    def __init__(self, input_filters, output_filters):
        super(CustomResidualUnit, self).__init__()
        self.pointwise1 = nn.Conv1d(input_filters, input_filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm1 = nn.BatchNorm1d(input_filters)
        self.depthwise_conv = nn.Conv1d(input_filters, input_filters, kernel_size=5, stride=1, padding=2, groups=input_filters, bias=False)
        self.norm2 = nn.BatchNorm1d(input_filters)
        self.pointwise2 = nn.Conv1d(input_filters, output_filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm3 = nn.BatchNorm1d(output_filters)
        self.act = nn.GELU()
        self.use_shortcut = input_filters == output_filters

    def forward(self, x):
        shortcut = x
        out = self.pointwise1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.depthwise_conv(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.pointwise2(out)
        out = self.norm3(out)
        if self.use_shortcut:
            out += shortcut
        return self.act(out)

# TabularFeatureExtractor
class TabularFeatureExtractor(nn.Module):
    def __init__(self, num_features=10, num_outputs=2, dropout_prob=0.3):
        super(TabularFeatureExtractor, self).__init__()
        self.dense1 = nn.Linear(num_features, 256)
        self.conv_initial = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.norm_initial = nn.BatchNorm1d(32, track_running_stats=False)
        self.act_initial = nn.GELU()
        self.res_unit1 = CustomResidualUnit(32, 32)
        self.res_unit2 = CustomResidualUnit(32, 64)
        self.conv_final = nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1)
        self.norm_final = nn.BatchNorm1d(1)
        self.act_final = nn.GELU()
        self.pooling = nn.MaxPool1d(kernel_size=3, stride=3)
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.dense2 = nn.Linear(85, 48)
        self.dense3 = nn.Linear(48, num_outputs)

    def forward(self, x):
        x = self.dense1(x).unsqueeze(1)
        x = self.conv_initial(x)
        x = self.norm_initial(x)
        x = self.act_initial(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.conv_final(x)
        x = self.norm_final(x)
        x = self.act_final(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.gelu(self.dense2(x))
        x = self.dropout_layer(x)
        x = self.dense3(x)
        return x

# Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and preprocessing objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TabularFeatureExtractor(num_features=25, num_outputs=2).to(device)
model.load_state_dict(torch.load("tabular_feature_extractor.pth"))
model.eval()
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_url():
    try:
        url = request.form.get('url')
        if not url:
            return render_template('result.html', error="No URL provided", success=False)
        if not is_valid_url(url):
            return render_template('result.html', error="Invalid URL format", success=False)

        features = extract_features(url)
        input_array = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        input_pca = pca.transform(input_scaled)
        input_tensor = torch.tensor(input_pca, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, prediction = torch.max(output, 1)
            prob_values = probabilities.cpu().numpy()[0].tolist()
            pred_value = int(prediction.cpu().numpy()[0])
            prediction_label = "Legitimate" if pred_value == 0 else "Phishing"

        return render_template(
            'result.html',
            url=url,
            prediction=prediction_label,
            probabilities=prob_values,
            success=True
        )
    except Exception as e:
        logging.error(f"Error during classification: {e}")
        return render_template('result.html', error=str(e), success=False)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return render_template('result.html', error="No file uploaded", success=False)

        file = request.files['file']
        if file.filename == '':
            return render_template('result.html', error="No file selected", success=False)

        urls = []
        if file.filename.endswith('.txt'):
            urls = file.read().decode('utf-8').splitlines()
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            if 'url' not in df.columns:
                return render_template('result.html', error="CSV file must contain a 'url' column", success=False)
            urls = df['url'].tolist()
        else:
            return render_template('result.html', error="Unsupported file format. Use .txt or .csv", success=False)

        results = {}
        for url in urls:
            if not is_valid_url(url):
                results[url] = "Invalid URL"
                continue
            features = extract_features(url)
            input_array = np.array(features).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            input_pca = pca.transform(input_scaled)
            input_tensor = torch.tensor(input_pca, dtype=torch.float32).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                _, prediction = torch.max(output, 1)
                pred_value = int(prediction.cpu().numpy()[0])
                prediction_label = "Legitimate" if pred_value == 0 else "Phishing"
                results[url] = prediction_label

        # Ensure static directory exists
        os.makedirs(os.path.join(app.root_path, 'static'), exist_ok=True)

        # Save results to a CSV file
        output_file = os.path.join(app.root_path, 'static', 'results.csv')
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['URL', 'Prediction'])
            for url, result in results.items():
                writer.writerow([url, result])

        return render_template(
            'result.html',
            batch_results=results,
            success=True,
            download_available=True
        )
    except Exception as e:
        logging.error(f"Error during batch prediction: {e}")
        return render_template('result.html', error=str(e), success=False)

@app.route('/download_results', methods=['GET'])
def download_results():
    output_file = os.path.join(app.root_path, 'static', 'results.csv')
    if os.path.exists(output_file):
        return send_file(output_file, as_attachment=True, download_name='results.csv')
    else:
        return render_template('result.html', error="No results file available for download", success=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)