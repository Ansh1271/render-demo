import os
import torch
import pickle
import gdown
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# --------- Settings ---------
MODEL_PATH = "fine_tuned_bert_model.pkl"
# Google Drive file ID of your 418MB model
DRIVE_FILE_ID = "1K0rAqhEzBj_CwLvVvBSbKufTj8phML4z"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# Label mapping for predictions
label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# --------- Download & Load Model ---------
def download_model():
    """Download the model from Google Drive if not already present."""
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model already exists at {MODEL_PATH}")
        return
    print("⬇️ Downloading model from Google Drive…")
    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("❌ Model download failed.")

try:
    download_model()
    with open(MODEL_PATH, "rb") as f:
        model: BertForSequenceClassification = pickle.load(f)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model.eval()
    print("✅ Model loaded from pickle.")
except Exception as e:
    model = None
    tokenizer = None
    print(f"⚠️ Error loading model: {e}")

# --------- Routes ---------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model/tokenizer not loaded on server"}), 500

    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No input text provided"}), 400

        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        # Run model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=-1).item()

        return jsonify({
            "input": text,
            "prediction": label_map[predicted_class_id]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------- Entry Point ---------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
