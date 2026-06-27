"""
GPT-2 Text Generator - Flask Application
Hosted on Hugging Face Spaces (runs the model directly, no API calls needed).
"""

from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the text-generation pipeline once at startup.
# HF Spaces provides enough RAM to run DistilGPT-2 comfortably.
# ---------------------------------------------------------------------------
print("Loading DistilGPT-2 pipeline...")
generator = pipeline("text-generation", model="distilgpt2")
print("Model ready.")


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
        result = generator(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.85,
            top_p=0.92,
            repetition_penalty=1.2,
            pad_token_id=50256,   # GPT-2 EOS token id
        )
        # pipeline returns full text; strip the original prompt off the front
        full_text = result[0]["generated_text"]
        generated = full_text[len(prompt):].strip()
        return jsonify({"generated_text": generated})

    except Exception as e:
        print(f"Generation error: {e}")
        return jsonify({"error": "Generation failed. Please try again."}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # HF Spaces default port
    app.run(host="0.0.0.0", port=port, debug=False)
