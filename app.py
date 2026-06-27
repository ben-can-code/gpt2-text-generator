"""
GPT-2 Text Generator - Flask Application
Uses Hugging Face transformers library with GPT-2 model to generate text continuations.
"""

from flask import Flask, render_template, request, jsonify
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model Loading
# Load the GPT-2 model and tokenizer once at startup to avoid reloading on
# every request. Using the small 'gpt2' checkpoint for fast inference.
# ---------------------------------------------------------------------------
print("Loading GPT-2 model... (this may take a moment on first run)")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()  # Set to evaluation mode (disables dropout, etc.)

# Use GPU if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"Model loaded successfully. Running on: {device.upper()}")


def generate_text(prompt: str, max_new_tokens: int = 100) -> str:
    """
    Generate a text continuation for the given prompt using GPT-2.

    Args:
        prompt: The input text to continue.
        max_new_tokens: Approximate number of new tokens to generate (~100 words).

    Returns:
        The generated continuation text (excluding the original prompt).
    """
    # Encode the prompt into token IDs
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate tokens with sampling for more natural, varied output
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,          # Enable sampling (vs greedy decoding)
            temperature=0.85,        # Controls randomness; lower = more focused
            top_p=0.92,              # Nucleus sampling: keep top 92% probability mass
            top_k=50,                # Also limit to top-50 tokens at each step
            repetition_penalty=1.2,  # Discourage repetitive phrases
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt tokens)
    new_token_ids = output_ids[0][input_ids.shape[-1]:]
    generated = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return generated.strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """
    Handle text generation requests.
    Expects JSON body: { "prompt": "<user text>" }
    Returns JSON: { "generated_text": "..." } or { "error": "..." }
    """
    data = request.get_json(silent=True)

    # --- Input validation ---
    if not data or "prompt" not in data:
        return jsonify({"error": "No prompt provided."}), 400

    prompt = data["prompt"].strip()

    if not prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    if len(prompt) > 500:
        return jsonify({"error": "Prompt is too long. Please keep it under 500 characters."}), 400

    # --- Generate text ---
    try:
        generated = generate_text(prompt)
        return jsonify({"generated_text": generated})
    except Exception as e:
        print(f"Generation error: {e}")
        return jsonify({"error": "Text generation failed. Please try again."}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Use PORT env variable if set (Render sets this automatically)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
