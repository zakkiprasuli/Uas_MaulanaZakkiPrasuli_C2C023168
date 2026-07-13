"""
Flask Web Application
=======================
Aplikasi web untuk mendemonstrasikan pipeline paper:
"Modified Histogram Equalization for Improved CNN Medical Image Segmentation"

Endpoint:
  GET  /                      -> UI utama
  POST /api/enhance           -> jalankan HE/CLAHE/HE-CLAHE/CLAHE-HE pada citra upload
  POST /api/segment           -> jalankan segmentasi CNN (U-Net) pada citra yang telah diproses
  GET  /api/model_summary     -> info arsitektur model (Fig. 6)
"""

import os
import sys
import glob
import base64
import io

import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhancement import apply_method, to_8bit, compute_histogram, compute_pdf, compute_cdf  # noqa: E402
from metrics import dice_similarity_coefficient, ssim_score  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

IMG_SIZE = 256

_MODEL_CACHE = {}


def get_model(method: str):
    """Muat model terlatih untuk sebuah metode enhancement jika tersedia
    di folder models/. Jika tidak ada, kembalikan None (mode arsitektur-saja)."""
    if method in _MODEL_CACHE:
        return _MODEL_CACHE[method]

    path = os.path.join(MODELS_DIR, f"{method}.h5")
    if os.path.exists(path):
        from tensorflow.keras.models import load_model
        model = load_model(path, compile=False)
    else:
        model = None
    _MODEL_CACHE[method] = model
    return model


def decode_upload(file_storage) -> np.ndarray:
    data = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("File citra tidak dapat dibaca.")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def encode_png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Gagal encode citra.")
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


def plot_hist_cdf_b64(hist, cdf, title) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.4), dpi=110)
    fig.patch.set_alpha(0)
    for ax in axes:
        ax.set_facecolor("none")
        ax.tick_params(colors="#8aa0b4", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#2c3e50")

    axes[0].bar(np.arange(len(hist)), hist, color="#3fd0c9", width=1.0)
    axes[0].set_title("Histogram", color="#e6f1ff", fontsize=9)

    axes[1].plot(np.arange(len(cdf)), cdf, color="#ff8a5c", linewidth=1.6)
    axes[1].set_title("CDF", color="#e6f1ff", fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


@app.route("/")
def index():
    available_models = [
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(MODELS_DIR, "*.h5"))
    ]
    return render_template("index.html", available_models=available_models)


@app.route("/api/enhance", methods=["POST"])
def api_enhance():
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file citra."}), 400

    try:
        img_raw = decode_upload(request.files["image"])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    cdf_min = int(request.form.get("cdf_min", 0))
    cdf_max = int(request.form.get("cdf_max", 39))
    clip_limit = float(request.form.get("clip_limit", 0.01))

    img8 = to_8bit(img_raw)
    img8 = cv2.resize(img8, (IMG_SIZE, IMG_SIZE))

    methods = ["original", "he", "clahe", "he_clahe", "clahe_he"]
    results = {}
    for m in methods:
        processed = apply_method(m, img8, cdf_min=cdf_min, cdf_max=cdf_max, clip_limit=clip_limit)
        hist = compute_histogram(processed)
        pdf = compute_pdf(hist)
        cdf = compute_cdf(pdf) * processed.size

        results[m] = {
            "image": encode_png_b64(processed),
            "hist_cdf_plot": plot_hist_cdf_b64(hist, cdf, m),
        }

    return jsonify({
        "results": results,
        "params": {"cdf_min": cdf_min, "cdf_max": cdf_max, "clip_limit": clip_limit},
    })


@app.route("/api/segment", methods=["POST"])
def api_segment():
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file citra."}), 400

    method = request.form.get("method", "clahe_he")
    cdf_min = int(request.form.get("cdf_min", 0))
    cdf_max = int(request.form.get("cdf_max", 39))
    clip_limit = float(request.form.get("clip_limit", 0.01))

    try:
        img_raw = decode_upload(request.files["image"])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    img8 = to_8bit(img_raw)
    img8 = cv2.resize(img8, (IMG_SIZE, IMG_SIZE))
    processed = apply_method(method, img8, cdf_min=cdf_min, cdf_max=cdf_max, clip_limit=clip_limit)

    model = get_model(method)
    model_status = "trained"

    if model is None:
        # Tidak ada model terlatih tersedia (belum training pada dataset nyata).
        # Bangun arsitektur untuk menunjukkan pipeline forward-pass berjalan,
        # namun tandai statusnya sebagai untrained agar transparan ke pengguna.
        from model import build_unet
        model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1))
        model_status = "untrained_architecture_demo"

    x = processed.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=(0, -1))
    pred = model.predict(x, verbose=0)[0, ..., 0]
    mask = (pred * 255).astype(np.uint8)

    overlay = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    mask_color = np.zeros_like(overlay)
    mask_color[..., 1] = mask  # highlight hijau pada region tersegmentasi
    overlay = cv2.addWeighted(overlay, 0.7, mask_color, 0.5, 0)

    response = {
        "model_status": model_status,
        "processed_image": encode_png_b64(processed),
        "mask": encode_png_b64(mask),
        "overlay": encode_png_b64(overlay),
    }

    if "ground_truth" in request.files:
        gt = decode_upload(request.files["ground_truth"])
        gt = cv2.resize(gt, (IMG_SIZE, IMG_SIZE))
        response["dsc"] = dice_similarity_coefficient(gt, mask)
        response["ssim"] = ssim_score(gt, mask)

    return jsonify(response)


@app.route("/api/model_summary")
def api_model_summary():
    from model import build_unet
    model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1))
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return jsonify({
        "summary": "\n".join(lines),
        "total_params": int(model.count_params()),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
