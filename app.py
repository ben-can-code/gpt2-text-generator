"""
GPT-2 Text Generator - Flask Application
Uses Hugging Face Inference API (no local model, no torch required).
"""

from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Hugging Face Inference API setup
# Get a free token at https://huggingface.co/settings/tokens
# ---------------------------------------------------------------------------
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_API_URL = "https://api-inference.huggingface.co/models/gpt2"

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}


def generate_text(prompt: str) -> str:
    """Call the HF Inference API and return the generated continuation."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.85,
            "top_p": 0.92,
            "do_sample": True,
            "return_full_text": False,   # return only the new text, not the prompt
        }
    }
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()

    # API returns a list: [{"generated_text": "..."}]
    if isinstance(result, list) and result:
        return result[0].get("generated_text", "").strip()

    raise ValueError("Unexpected response from API")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True)

    if not data or not data.get("prompt", "").strip():
        return jsonify({"error": "Prompt cannot be empty."}), 400

    prompt = data["prompt"].strip()

    if len(prompt) > 500:
        return jsonify({"error": "Prompt is too long. Keep it under 500 characters."}), 400

    try:
        generated = generate_text(prompt)
        return jsonify({"generated_text": generated})
    except Exception as e:
        print(f"API error: {e}")
        return jsonify({"error": "Generation failed. Please try again in a moment."}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
