import os
import torch
import pickle
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Label mapping
label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Load tokenizer & model
MODEL_PATH = "fine_tuned_bert_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model: BertForSequenceClassification = pickle.load(f)
    print("✅ Model loaded from pickle")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model.eval()
except Exception as e:
    model = None
    tokenizer = None
    print(f"⚠️ Error loading model: {e}")

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)