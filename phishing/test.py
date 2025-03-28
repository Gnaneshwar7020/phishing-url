import torch
import torch.nn as nn
import numpy as np
import joblib
import argparse
import re
import tldextract
from urllib.parse import urlparse
from collections import Counter
import nltk
from nltk.corpus import words

# Download NLTK words if not already downloaded
nltk.download('words', quiet=True)

# Define brand names and suspicious TLDs (as provided)
BRAND_NAMES = set([
    "google", "facebook", "amazon", "apple", "microsoft", "twitter", "instagram",
    "linkedin", "paypal", "dropbox", "alibaba", "ebay", "netflix", "youtube",
    "wordpress", "adobe", "cisco", "oracle", "salesforce", "shopify", "walmart",
    "target", "bestbuy", "costco", "tesla", "uber", "airbnb", "booking", "expedia",
    "spotify", "slack", "zoom", "tiktok", "snapchat", "reddit", "pinterest",
    "whatsapp", "telegram", "discord", "skype", "samsung", "sony", "nike", "adidas",
    "nvidia", "intel", "hp", "dell", "lenovo", "asus", "huawei", "xiaomi", "oneplus",
    "logitech", "seagate", "western digital", "sandisk", "canon", "nikon", "panasonic",
    "philips", "siemens", "bmw", "mercedes", "toyota", "honda", "ford", "volkswagen",
    "chevrolet", "nissan", "hyundai", "kia", "pepsi", "cocacola", "starbucks",
    "mcdonalds", "burgerking", "subway", "dominos", "pizzahut", "kfc", "nintendo",
    "playstation", "xbox", "activision", "ea", "riotgames", "blizzard", "unity",
    "godaddy", "namecheap", "bluehost", "hostgator", "siteground", "cloudflare",
    "akamai", "verisign", "symantec", "mcafee", "avast", "avg", "bitdefender",
    "kaspersky", "norton", "zendesk", "freshdesk", "mailchimp", "sendgrid", "hubspot",
    "hootsuite", "buffer", "semrush", "ahrefs", "moz", "glassdoor", "indeed", "monster",
    "ziprecruiter", "udemy", "coursera", "edx", "khanacademy", "wikipedia", "quora",
    "stackoverflow", "github", "gitlab", "docker", "kubernetes", "jenkins", "ansible",
    "hashicorp", "atlassian"
])

SUSPICIOUS_TLDS = set([
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "online", "site",
    "info", "biz", "cc", "pw", "click", "link", "win", "faith", "party",
    "date", "loan", "stream", "download", "review", "bid", "trade", "accountant",
    "country", "science", "men", "gdn", "racing", "webcam", "video", "chat",
    "mom", "work", "life", "live", "tech", "space", "website", "press", "rest",
    "market", "pub", "social", "rocks", "world", "city", "today", "company",
    "zone", "wiki", "support", "solutions", "email", "network", "center", "pro",
    "store", "shop", "services", "industries", "directory", "foundation",
    "international", "systems", "vision", "community", "cash", "fund", "marketing",
    "media", "money", "name", "report", "school", "eco", "one", "digital", "ltd",
    "group", "institute", "academy", "courses", "events", "agency", "technology",
    "management", "consulting", "properties", "rentals", "vacations", "careers",
    "education", "financial", "healthcare", "house", "land", "mortgage", "energy",
    "engineering", "recipes", "tips", "tools", "training", "university", "villas",
    "airforce", "army", "bargains", "blackfriday", "blue", "build", "builders",
    "cards", "cheap", "christmas", "claims", "cleaning", "clinic", "clothing",
    "coach", "codes", "coffee", "condos", "construction", "contractors", "coupons",
    "credit", "cricket", "dating", "deals", "delivery", "democrat", "dental",
    "discount", "dog", "domains", "equipment", "estate", "exchange", "exposed",
    "fail", "farm", "finance", "fishing", "fit", "flights", "florist", "football",
    "forsale", "furniture", "gallery", "games", "gifts", "glass", "gold", "golf",
    "graphics", "green", "haus", "health", "hockey", "holdings", "holiday",
    "immobilien", "industries", "insure", "kim", "kitchen", "lawyer", "lease",
    "legal", "lgbt", "limited", "limo", "loan", "luxury", "maison", "moda",
    "monster", "nagoya", "ninja", "ong", "organic", "partners", "parts", "photo",
    "photography", "pics", "pictures", "pink", "pizza", "place", "plumbing", "plus"
])

# Feature extraction function (as provided)
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

# CustomResidualUnit (unchanged)
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

# TabularFeatureExtractor (unchanged)
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

# Load model and preprocessing objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TabularFeatureExtractor(num_features=10, num_outputs=2).to(device)
model.load_state_dict(torch.load("model/tabular_feature_extractor.pth"))
model.eval()

scaler = joblib.load("model/scaler.pkl")
pca = joblib.load("model/pca.pkl")

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Test the TabularFeatureExtractor model with a URL.")
parser.add_argument("url", type=str, help="URL to classify (e.g., 'https://example.com')")
args = parser.parse_args()

# Extract features from the URL
try:
    features = extract_features(args.url)
    if len(features) != 55:
        raise ValueError("Feature extraction failed to produce 55 features.")
except Exception as e:
    print(f"Error extracting features: {e}")
    print("Please provide a valid URL.")
    exit(1)

# Preprocess input
input_array = np.array(features).reshape(1, -1)  # Shape: (1, 55)
input_scaled = scaler.transform(input_array)
input_pca = pca.transform(input_scaled)

# Convert to tensor
input_tensor = torch.tensor(input_pca, dtype=torch.float32).to(device)

# Make prediction
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    _, prediction = torch.max(output, 1)
    print(f"Probabilities: {probabilities.cpu().numpy()[0]}")
    print(f"Predicted class: {prediction.cpu().numpy()[0]}")