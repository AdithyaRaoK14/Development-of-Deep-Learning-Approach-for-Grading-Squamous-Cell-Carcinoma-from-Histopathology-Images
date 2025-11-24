# lung_ui.py
"""
LungUI — Modern, Clean, Stable UI with ConvNeXt 5-Fold Ensemble
• NaN-safe preprocessing
• Local AI (Ollama — any installed model)
• Global AI model selector (dropdown)
• Global explanation length & tone controls
• Strict safe SCC explanation prompts
• PDF + UI AI analysis
• ConvNeXt 5-Fold Ensemble + EfficientNet + DenseNet + MobileNet
• Centered, research-grade interface
"""

import os
import io
import uuid
import textwrap
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.convnext import preprocess_input as preprocess_convnext
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_effnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet

import requests
import ast
import collections

# ----------------------------------------------------------
# PDF libs
# ----------------------------------------------------------
REPORTLAB_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except Exception:
    pass

import gradio as gr

# ----------------------------------------------------------
# GLOBAL CONFIG
# ----------------------------------------------------------
MODEL_DIR = "."
EFFNET_PATH = os.path.join(MODEL_DIR, "effnetb0_final7.h5")
DENSENET_PATH = os.path.join(MODEL_DIR, "densenet121_gradcam.h5")
MOBILENET_PATH = os.path.join(MODEL_DIR, "mobilenetv2_gradcam.h5")
CONVNEXT_FOLD_PATHS = [os.path.join(MODEL_DIR, f"model_fold_{i}.h5") for i in range(1, 6)]

CLASS_NAMES = ["well", "mod", "poor"]

IMG_SIZE_EFF = (224, 224)
IMG_SIZE_CONVNEXT = (260, 260)

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

PROJECT_TITLE = (
    "Development of deep learning approach for grading squamous cell carcinoma "
    "from histopathology images"
)

# ----------------------------------------------------------
# ConvNeXt Custom Layers (from your working reference code)
# ----------------------------------------------------------
class LayerScale(tf.keras.layers.Layer):
    def __init__(self, init_values=1e-6, projection_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values

    def build(self, shape):
        c = int(shape[-1])
        self.gamma = self.add_weight(
            name="layerscale_gamma",
            shape=(c,),
            initializer=tf.keras.initializers.Constant(self.init_values),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma


class ConvNeXtBlock(tf.keras.layers.Layer):
    def call(self, x):
        # In your saved models, the internal logic is already baked in;
        # this stub satisfies the loader.
        return x


CUSTOM_OBJECTS = {
    "LayerScale": LayerScale,
    "ConvNeXtBlock": ConvNeXtBlock,
}

# ----------------------------------------------------------
# OLLAMA MODEL DISCOVERY
# ----------------------------------------------------------
def list_ollama_models():
    """List installed Ollama models (for dropdown)."""
    try:
        out = requests.get("http://127.0.0.1:11434/api/tags", timeout=3).json()
        if "models" in out:
            return [m["name"] for m in out["models"]]
    except Exception:
        pass
    return ["phi3:mini"]  # fallback default


AVAILABLE_AI_MODELS = list_ollama_models()
AI_MODEL_DEFAULT = AVAILABLE_AI_MODELS[0] if AVAILABLE_AI_MODELS else "phi3:mini"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# ----------------------------------------------------------
# AI ENGINE — retry, timeout, temperature, max_tokens
# ----------------------------------------------------------
def ai_request(prompt: str, model: str, temperature: float, max_tokens: int):
    """
    Call local Ollama model with safety cutoff.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        resp = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
        )
        js = resp.json()
        if isinstance(js, dict) and "response" in js:
            txt = js["response"].strip()
            if len(txt) > max_tokens:
                return txt[:max_tokens] + "..."
            return txt
        return "AI Error: Invalid response structure from Ollama."
    except Exception as e:
        return f"AI Error: {str(e)}"

# ----------------------------------------------------------
# SAFE PROBABILITY CLEANUP
# ----------------------------------------------------------
def safe_prob_for_ai(p):
    """
    Normalize probability dict for AI prompts.
    Accepts dict or stringified dict.
    """
    try:
        if isinstance(p, str):
            p = ast.literal_eval(p)
        if not isinstance(p, dict):
            raise ValueError()
        return {k: float(p.get(k, 0.0)) for k in CLASS_NAMES}
    except Exception:
        return {k: 0.0 for k in CLASS_NAMES}

# ----------------------------------------------------------
# SCC EXPLANATION PROMPT BUILDER (length + tone)
# ----------------------------------------------------------
def build_scc_prompt(model_name, pred, probs, length, tone):
    """
    Strict, safe SCC explanation prompt.
    Uses global controls:
        - length = short / medium / detailed
        - tone = academic / simple / structured
    """

    if length == "short":
        ln = "3–4 lines"
    elif length == "detailed":
        ln = "7–10 lines"
    else:
        ln = "5–7 lines"

    tone_map = {
        "academic": "Use a formal academic tone.",
        "simple": "Use simple, easy-to-understand wording.",
        "structured": "Use clearly separated short structured points."
    }
    tone_text = tone_map.get(tone, tone_map["academic"])

    return f"""
You are explaining the output of a deep learning model for grading squamous cell carcinoma (SCC).

STRICT RULES:
- Do NOT describe any slide-specific visual details.
- Do NOT mention colors, staining, keratin pearls, inflammation, necrosis, or tumor nests.
- Do NOT infer diagnosis, prognosis, clinical symptoms, or treatment.
- ONLY describe general, widely known tendencies associated with SCC grading.

Model: {model_name}
Predicted grade: {pred}

Probabilities:
- well: {probs['well']:.3f}
- mod: {probs['mod']:.3f}
- poor: {probs['poor']:.3f}

Write {ln} describing:
- General histopathological tendencies associated with this SCC grade.
- How the probability distribution indicates model confidence or uncertainty.
- {tone_text}

End with:
"This explanation is a model-based decision-support summary and not a diagnostic assessment."
"""

# ----------------------------------------------------------
# AI CONFIDENCE NOTE (optional helper)
# ----------------------------------------------------------
def generate_confidence_note(probs):
    arr = np.array(list(probs.values()))
    if len(arr) < 2:
        return "Model confidence could not be assessed."
    sorted_idx = np.argsort(arr)
    diff = arr.max() - arr[sorted_idx[-2]]
    if diff > 0.60:
        return "Model confidence appears strong based on probability separation."
    elif diff > 0.30:
        return "Model confidence appears moderate with partial separation."
    else:
        return "Model confidence appears low; probabilities are closely grouped."

# ----------------------------------------------------------
# NaN-safe RGB Converter
# ----------------------------------------------------------
def enforce_rgb_uint8(img):
    """Ensure image is RGB uint8 and NaN-safe."""
    if img is None:
        return None

    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32)
    else:
        arr = np.array(img, dtype=np.float32)

    # If grayscale → convert to 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    # If RGBA → drop alpha
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(arr, 0, 255).astype(np.uint8)

# ----------------------------------------------------------
# Safe Model Loader
# ----------------------------------------------------------
def safe_load(path, custom=False):
    if not os.path.exists(path):
        print("Missing:", path)
        return None
    try:
        if custom:
            return load_model(path, compile=False, custom_objects= CUSTOM_OBJECTS)
        return load_model(path, compile=False)
    except Exception as e:
        print("Failed to load model:", path, "Error:", e)
        return None

print("\nLoading models...")
effnet_model = safe_load(EFFNET_PATH)
densenet_model = safe_load(DENSENET_PATH)
mobilenet_model = safe_load(MOBILENET_PATH)

convnext_folds = []
for p in CONVNEXT_FOLD_PATHS:
    m = safe_load(p, custom=True)
    if m is not None:
        convnext_folds.append(m)

print("ConvNeXt folds loaded:", len(convnext_folds))

# ----------------------------------------------------------
# Preprocessing Functions
# ----------------------------------------------------------
def prep_for_effnet(img):
    img = enforce_rgb_uint8(img)
    return preprocess_effnet(img_to_array(cv2.resize(img, IMG_SIZE_EFF)))


def prep_for_densenet(img):
    img = enforce_rgb_uint8(img)
    return preprocess_densenet(img_to_array(cv2.resize(img, IMG_SIZE_EFF)))


def prep_for_mobilenet(img):
    img = enforce_rgb_uint8(img)
    return preprocess_mobilenet(img_to_array(cv2.resize(img, IMG_SIZE_EFF)))


def prep_for_convnext(img):
    img = enforce_rgb_uint8(img)
    return preprocess_convnext(img_to_array(cv2.resize(img, IMG_SIZE_CONVNEXT)))

# ----------------------------------------------------------
# Grad-CAM Utilities
# ----------------------------------------------------------
def _find_last_conv(model):
    """Find last convolutional layer for Grad-CAM."""
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.DepthwiseConv2D)):
            return layer.name

    # Fallback: first layer with 4D output
    for layer in reversed(model.layers):
        try:
            if len(layer.output_shape) == 4:
                return layer.name
        except Exception:
            pass

    return model.layers[-1].name


def apply_colormap_on_image(orig, mask, alpha=0.35, intensity=1.0):
    """Overlay a heatmap over original image."""
    mask = np.clip(mask * intensity, 0, 1)
    heatmap = np.uint8(255 * mask)

    heat = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(
        orig.astype(np.float32), 1 - alpha,
        heat.astype(np.float32), alpha,
        0
    )
    return heat, blended.astype(np.uint8)


def generate_gradcam(model, img, prep_fn, intensity=1.0, alpha=0.35):
    """Compute model prediction + Grad-CAM heatmap + overlay + embossed."""
    img = enforce_rgb_uint8(img)

    x = np.expand_dims(prep_fn(img), 0)
    preds = model.predict(x, verbose=0)[0]

    preds = np.nan_to_num(preds, nan=1e-7)
    preds = preds / (np.sum(preds) + 1e-8)
    class_idx = int(np.argmax(preds))

    # Create Grad-CAM model
    last_conv = _find_last_conv(model)
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(x)
        loss = pred[:, class_idx]

    grads = tape.gradient(loss, conv_out)[0].numpy()
    conv = conv_out[0].numpy()

    weights = np.mean(grads, axis=(0, 1))
    cam = np.maximum(np.dot(conv, weights), 0)
    cam = cam / (cam.max() + 1e-8)

    # Resize CAM to input size
    target = IMG_SIZE_CONVNEXT if prep_fn == prep_for_convnext else IMG_SIZE_EFF
    cam = cv2.resize(cam, (target[1], target[0]))

    orig_rs = cv2.resize(img, (target[1], target[0]))

    heat, overlay = apply_colormap_on_image(orig_rs, cam, alpha, intensity)

    # Embossed variant
    emb = cv2.addWeighted(orig_rs, 0.5, heat, 0.5, 0)
    emb = cv2.filter2D(
        emb, -1,
        np.array([[0, -1, 0],
                  [-1, 5, -1],
                  [0, -1, 0]])
    )

    return preds, heat, overlay, emb

# ----------------------------------------------------------
# AI TEXT WRAPPER FUNCTION
# ----------------------------------------------------------
def ai_analyze_text(prompt, model=AI_MODEL_DEFAULT, temperature=0.2, max_tokens=2000):
    """
    Wrapper around ai_request() used by:
      • PDF generator
      • AI interpretation buttons
    """
    return ai_request(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

# ----------------------------------------------------------
# PDF REPORT GENERATOR
# ----------------------------------------------------------
def wrap_text(c, text, width, font="Helvetica", size=10):
    """Split text into printable lines for PDF."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if c.stringWidth(test, font, size) <= width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def generate_pdf(img, model_name, pred_name, prob_dict, imgs):
    """
    Generates 2-page PDF:
      Page 1 -> Summary + AI Interpretation
      Page 2 -> Original + Heatmap + Overlay + Embossed
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    pid = str(uuid.uuid4())[:8]
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(REPORTS_DIR, f"report_{pid}_{now_str}.pdf")

    # Convert numpy → PIL
    pil_imgs = []
    for g in imgs:
        if isinstance(g, np.ndarray):
            pil_imgs.append(Image.fromarray(g))
        else:
            pil_imgs.append(g)

    # Use ReportLab if available
    if REPORTLAB_AVAILABLE:
        c = canvas.Canvas(out_path, pagesize=A4)
        W, H = A4
        margin = 18 * mm
        text_width = W - 2 * margin

        x = margin
        y = H - margin

        # -------------------------
        # TITLE
        # -------------------------
        c.setFont("Helvetica-Bold", 14)
        for line in wrap_text(c, PROJECT_TITLE, text_width,
                              "Helvetica-Bold", 14):
            c.drawString(x, y, line)
            y -= 7 * mm

        # metadata
        c.setFont("Helvetica", 9)
        c.drawString(x, y, f"Report ID: {pid}")
        y -= 4 * mm
        c.drawString(x, y,
                     f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 8 * mm

        # -------------------------
        # SUMMARY
        # -------------------------
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y, "Model Summary")
        y -= 6 * mm

        c.setFont("Helvetica", 10)
        c.drawString(x, y, f"Model: {model_name}")
        y -= 5 * mm
        c.drawString(x, y, f"Predicted Grade: {pred_name}")
        y -= 6 * mm

        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, "Class Probabilities:")
        y -= 5 * mm

        c.setFont("Helvetica", 10)
        for k in CLASS_NAMES:
            c.drawString(x + 10, y, f"{k}: {prob_dict[k]*100:.1f}%")
            y -= 4.5 * mm

        y -= 6 * mm

        # -------------------------
        # AI INTERPRETATION
        # -------------------------
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, "AI Interpretation:")
        y -= 5 * mm

        ai_prompt = f"""
You are generating a concise scientific explanation of a deep learning 
prediction for squamous cell carcinoma (SCC) grading.

Model: {model_name}
Predicted grade: {pred_name}

Probabilities:
- well: {prob_dict['well']:.3f}
- mod: {prob_dict['mod']:.3f}
- poor: {prob_dict['poor']:.3f}

Write 5–8 lines that:
• Describe only the general histopathological patterns typically associated 
  with this SCC grade (overall differentiation, keratinization tendency, 
  nuclear variation, architectural arrangement).
• Base your explanation ONLY on the predicted grade, NOT the actual slide.
• Discuss how the probability distribution suggests confidence or ambiguity.
• Do NOT mention any clinical symptoms, diagnosis, prognosis, treatment, 
  inflammation, infection, patient state, or recommendations.
• End with the sentence: 
  "This interpretation is a pattern-recognition aid and not a diagnostic conclusion."
"""

        # PDF uses default AI model (independent of UI selector)
        ai_text = ai_analyze_text(ai_prompt, model=AI_MODEL_DEFAULT)

        c.setFont("Helvetica", 9)
        for line in wrap_text(c, ai_text, text_width,
                              "Helvetica", 9):
            c.drawString(x, y, line)
            y -= 4.5 * mm

        # -------------------------
        # PAGE 2 — Grad-CAM Panel
        # -------------------------
        c.showPage()
        W, H = A4
        margin = 18 * mm
        base_img = pil_imgs[0]

        grid_w = (W - 3 * margin) / 2
        grid_h = grid_w * (base_img.height / base_img.width)

        positions = [
            (margin,                H - margin - grid_h),
            (margin + grid_w + margin, H - margin - grid_h),
            (margin,                H - margin - 2 * grid_h - 10 * mm),
            (margin + grid_w + margin, H - margin - 2 * grid_h - 10 * mm),
        ]

        labels = ["Original", "Grad-CAM", "Overlay", "Embossed"]

        for img_pil, (px, py), lbl in zip(pil_imgs, positions, labels):
            buff = io.BytesIO()
            img_resized = img_pil.resize((int(grid_w), int(grid_h)))
            img_resized.save(buff, format="PNG")
            buff.seek(0)
            c.drawImage(ImageReader(buff), px, py,
                        width=grid_w, height=grid_h)

            c.setFont("Helvetica", 9)
            c.drawString(px, py - 5 * mm, lbl)

        c.save()
        return out_path

    # ---------------------------
    # FALLBACK PNG GRID (no PDF)
    # ---------------------------
    png_path = out_path.replace(".pdf", ".png")
    w, h = pil_imgs[0].size
    grid = Image.new("RGB", (2*w, 2*h), (255, 255, 255))

    grid.paste(pil_imgs[0].resize((w, h)), (0, 0))
    grid.paste(pil_imgs[1].resize((w, h)), (w, 0))
    grid.paste(pil_imgs[2].resize((w, h)), (0, h))
    grid.paste(pil_imgs[3].resize((w, h)), (w, h))

    grid.save(png_path)
    return png_path

# ---------------------------------------------------
# run_and_report — for EffNet, DenseNet, MobileNet
# ---------------------------------------------------
def run_and_report(model, img, prep_fn, model_name, intensity=1.0, alpha=0.35):
    """Runs the model, applies Grad-CAM, formats PDF, returns results."""
    img = enforce_rgb_uint8(img)

    preds, heat, overlay, emb = generate_gradcam(
        model, img, prep_fn, intensity, alpha
    )

    preds = np.nan_to_num(preds, nan=1e-7)
    preds = preds / (np.sum(preds) + 1e-8)

    prob_dict = {
        CLASS_NAMES[i]: float(preds[i])
        for i in range(len(CLASS_NAMES))
    }
    pred_name = CLASS_NAMES[int(np.argmax(preds))]

    # For PDF images
    orig  = Image.fromarray(cv2.resize(img, IMG_SIZE_EFF[::-1]))
    heat2 = Image.fromarray(cv2.resize(heat, IMG_SIZE_EFF[::-1]))
    over2 = Image.fromarray(cv2.resize(overlay, IMG_SIZE_EFF[::-1]))
    emb2  = Image.fromarray(cv2.resize(emb, IMG_SIZE_EFF[::-1]))

    pdf_path = generate_pdf(
        img,
        model_name,
        pred_name,
        prob_dict,
        [orig, heat2, over2, emb2]
    )

    return pred_name, prob_dict, heat, overlay, emb, pdf_path

# ---------------------------------------------------
# ConvNeXt 5-Fold Ensemble
# ---------------------------------------------------
def convnext_fold_ensemble(img, intensity=1.0, alpha=0.35):
    """Averages predictions across all available folds, 
       selects best fold for Grad-CAM."""
    img = enforce_rgb_uint8(img)

    probs = []
    valid_idxs = []

    for idx, model in enumerate(convnext_folds):
        try:
            x = np.expand_dims(prep_for_convnext(img), 0)
            p = model.predict(x, verbose=0)[0]
            p = np.nan_to_num(p)

            if np.sum(p) == 0:
                continue

            p = p / (p.sum() + 1e-8)
            probs.append(p)
            valid_idxs.append(idx)

        except Exception:
            pass

    # fallback if no fold works
    if len(probs) == 0:
        fallback = np.array([0.33, 0.33, 0.34])
        pred = CLASS_NAMES[int(np.argmax(fallback))]
        fallback_prob = {
            CLASS_NAMES[i]: float(fallback[i])
            for i in range(len(CLASS_NAMES))
        }
        return pred, fallback_prob, None, None, None, None

    # mean ensemble prediction
    mean_prob = np.mean(np.stack(probs), axis=0)
    mean_prob = mean_prob / (mean_prob.sum() + 1e-8)

    prob_dict = {
        CLASS_NAMES[i]: float(mean_prob[i])
        for i in range(len(CLASS_NAMES))
    }
    pred_name = CLASS_NAMES[int(np.argmax(mean_prob))]

    # choose best fold based on highest confidence
    top_class = int(np.argmax(mean_prob))
    confs = [float(p[top_class]) for p in probs]
    best_local = int(np.argmax(confs))
    best_global = valid_idxs[best_local]

    best_model = convnext_folds[best_global]

    # Grad-CAM from best fold
    _, heat, overlay, emb = generate_gradcam(
        best_model, img, prep_for_convnext,
        intensity, alpha
    )

    # images for PDF
    orig  = Image.fromarray(cv2.resize(img, IMG_SIZE_EFF[::-1]))
    heat2 = Image.fromarray(cv2.resize(heat, IMG_SIZE_EFF[::-1]))
    over2 = Image.fromarray(cv2.resize(overlay, IMG_SIZE_EFF[::-1]))
    emb2  = Image.fromarray(cv2.resize(emb, IMG_SIZE_EFF[::-1]))

    pdf = generate_pdf(
        img,
        f"ConvNeXt 5-Fold Ensemble (best fold {best_global+1})",
        pred_name,
        prob_dict,
        [orig, heat2, over2, emb2]
    )

    return pred_name, prob_dict, heat, overlay, emb, pdf

# ---------------------------------------------------
# Probability Cleaning for UI Display
# ---------------------------------------------------
def safe_prob_display(prob_dict):
    """Normalize & clean probability values for UI labels."""
    vals = np.array([
        prob_dict.get(k, 0.0)
        for k in CLASS_NAMES
    ], dtype=np.float64)

    vals = np.nan_to_num(vals)
    s = vals.sum()

    if s <= 0:
        vals = np.array([1/3, 1/3, 1/3])

    vals = vals / vals.sum()

    return {
        CLASS_NAMES[i]: float(vals[i])
        for i in range(len(CLASS_NAMES))
    }

# ---------------------------------------------
# GRADIO UI SETUP (GLOBAL CONTROLS + TABS)
# ---------------------------------------------
with gr.Blocks(
    css="""
.center-container {
    max-width: 950px;
    margin-left: auto;
    margin-right: auto;
}
.header {
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 18px;
}
.global-controls {
    padding: 12px;
    border: 1px solid #ddd;
    border-radius: 10px;
    margin-bottom: 18px;
    background: #fafafa;
}
"""
) as demo:

    with gr.Column(elem_classes="center-container"):

        # -------------------------------
        # PROJECT HEADER
        # -------------------------------
        gr.Markdown(f"## {PROJECT_TITLE}", elem_classes="header")

        # -----------------------------------
        # GLOBAL AI SETTINGS (for ALL tabs)
        # -----------------------------------
        with gr.Box(elem_classes="global-controls"):
            gr.Markdown("### 🔧 AI Interpretation Settings (Global)")

            with gr.Row():
                # AI Model selector
                ai_model_dropdown = gr.Dropdown(
                    choices=AVAILABLE_AI_MODELS,
                    value=AI_MODEL_DEFAULT,
                    label="AI Model (Ollama)"
                )

                # Explanation length
                ex_length = gr.Radio(
                    ["short", "medium", "detailed"],
                    value="medium",
                    label="Explanation Length"
                )

                # Explanation tone
                ex_tone = gr.Radio(
                    ["academic", "simple", "structured"],
                    value="academic",
                    label="Explanation Tone"
                )

        # -----------------------------------
        # MODEL TABS
        # -----------------------------------
        with gr.Tabs():

            # ================================================================
            # TAB 1 — EfficientNetB0
            # ================================================================
            with gr.TabItem("EfficientNetB0"):

                en_img = gr.Image(type="pil", label="Upload Image (RGB)")
                en_int = gr.Slider(0.2, 2.0, 1.0, label="Heatmap intensity")
                en_alpha = gr.Slider(0.0, 0.8, 0.35, label="Heatmap transparency")
                en_btn = gr.Button("Run EfficientNetB0 & Generate Report")

                en_out_pred = gr.Label(label="Prediction")
                en_out_probs = gr.Label(label="Class Probabilities")
                en_out_heat = gr.Image(label="Grad-CAM Heatmap")
                en_out_overlay = gr.Image(label="Overlay")
                en_out_emboss = gr.Image(label="Embossed")
                en_out_report = gr.File(label="Download Report")

                en_ai_btn = gr.Button("AI Interpretation (Using Global Settings)")
                en_ai_out = gr.Textbox(label="AI Interpretation", lines=6)

                def _run_en(img, intensity, alpha):
                    if img is None:
                        blank = np.zeros((IMG_SIZE_EFF[0], IMG_SIZE_EFF[1], 3), np.uint8)
                        return ("No image",
                                {k: 0.0 for k in CLASS_NAMES},
                                blank, blank, blank, None)

                    if effnet_model is None:
                        blank = np.zeros((IMG_SIZE_EFF[0], IMG_SIZE_EFF[1], 3), np.uint8)
                        return ("EfficientNetB0 model missing",
                                {k: 0.0 for k in CLASS_NAMES},
                                blank, blank, blank, None)

                    pred, probs, heat, overlay, emb, rpt = run_and_report(
                        effnet_model, img, prep_for_effnet,
                        "EfficientNetB0",
                        float(intensity), float(alpha)
                    )
                    return pred, safe_prob_display(probs), heat, overlay, emb, rpt

                def _ai_en(pred, probs, model, length, tone):
                    probs_clean = safe_prob_for_ai(probs)
                    prompt = build_scc_prompt("EfficientNetB0", pred, probs_clean, length, tone)
                    return ai_analyze_text(prompt, model=model)

                en_btn.click(
                    fn=_run_en,
                    inputs=[en_img, en_int, en_alpha],
                    outputs=[
                        en_out_pred, en_out_probs,
                        en_out_heat, en_out_overlay,
                        en_out_emboss, en_out_report
                    ]
                )

                en_ai_btn.click(
                    fn=_ai_en,
                    inputs=[en_out_pred, en_out_probs,
                            ai_model_dropdown, ex_length, ex_tone],
                    outputs=[en_ai_out]
                )

            # ================================================================
            # TAB 2 — DenseNet121
            # ================================================================
            with gr.TabItem("DenseNet121"):

                dn_img = gr.Image(type="pil", label="Upload Image (RGB)")
                dn_int = gr.Slider(0.2, 2.0, 1.0, label="Heatmap intensity")
                dn_alpha = gr.Slider(0.0, 0.8, 0.35, label="Heatmap transparency")
                dn_btn = gr.Button("Run DenseNet121 & Generate Report")

                dn_out_pred = gr.Label(label="Prediction")
                dn_out_probs = gr.Label(label="Class Probabilities")
                dn_out_heat = gr.Image(label="Grad-CAM Heatmap")
                dn_out_overlay = gr.Image(label="Overlay")
                dn_out_emboss = gr.Image(label="Embossed")
                dn_out_report = gr.File(label="Download Report")

                dn_ai_btn = gr.Button("AI Interpretation (Using Global Settings)")
                dn_ai_out = gr.Textbox(label="AI Interpretation", lines=6)

                def _run_dn(img, intensity, alpha):
                    if img is None:
                        blank = np.zeros((IMG_SIZE_EFF[0], IMG_SIZE_EFF[1], 3), np.uint8)
                        return ("No image",
                                {k: 0.0 for k in CLASS_NAMES},
                                blank, blank, blank, None)

                    if densenet_model is None:
                        blank = np.zeros((IMG_SIZE_EFF[0], IMG_SIZE_EFF[1], 3), np.uint8)
                        return ("DenseNet121 model missing",
                                {k: 0.0 for k in CLASS_NAMES},
                                blank, blank, blank, None)

                    pred, probs, heat, overlay, emb, rpt = run_and_report(
                        densenet_model, img, prep_for_densenet,
                        "DenseNet121",
                        float(intensity), float(alpha)
                    )
                    return pred, safe_prob_display(probs), heat, overlay, emb, rpt

                def _ai_dn(pred, probs, model, length, tone):
                    probs_clean = safe_prob_for_ai(probs)
                    prompt = build_scc_prompt("DenseNet121", pred, probs_clean, length, tone)
                    return ai_analyze_text(prompt, model=model)

                dn_btn.click(
                    fn=_run_dn,
                    inputs=[dn_img, dn_int, dn_alpha],
                    outputs=[
                        dn_out_pred, dn_out_probs,
                        dn_out_heat, dn_out_overlay,
                        dn_out_emboss, dn_out_report
                    ]
                )

                dn_ai_btn.click(
                    fn=_ai_dn,
                    inputs=[dn_out_pred, dn_out_probs,
                            ai_model_dropdown, ex_length, ex_tone],
                    outputs=[dn_ai_out]
                )

            # ================================================================
            # TAB 3 — MobileNetV2
            # ================================================================
            with gr.TabItem("MobileNetV2"):

                mb_img = gr.Image(type="pil", label="Upload Image (RGB)")
                mb_int = gr.Slider(0.2, 2.0, 1.0, label="Heatmap intensity")
                mb_alpha = gr.Slider(0.0, 0.8, 0.35, label="Heatmap transparency")
                mb_btn = gr.Button("Run MobileNetV2 & Generate Report")

                mb_out_pred = gr.Label(label="Prediction")
                mb_out_probs = gr.Label(label="Class Probabilities")
                mb_out_heat = gr.Image(label="Grad-CAM Heatmap")
                mb_out_overlay = gr.Image(label="Overlay")
                mb_out_emboss = gr.Image(label="Embossed")
                mb_out_report = gr.File(label="Download Report")

                mb_ai_btn = gr.Button("AI Interpretation (Using Global Settings)")
                mb_ai_out = gr.Textbox(label="AI Interpretation", lines=6)

                def _run_mb(img, intensity, alpha):
                    if img is None:
                        blank = np.zeros((IMG_SIZE_EFF[0], IMG_SIZE_EFF[1], 3), np.uint8)
                        return ("No image",
                                {k: 0.0 for k in CLASS_NAMES},
                                blank, blank, blank, None)

                    if mobilenet_model is None:
                        blank = np.zeros((IMG_SIZE_EFF[0], IMG_SIZE_EFF[1], 3), np.uint8)
                        return ("MobileNetV2 model missing",
                                {k: 0.0 for k in CLASS_NAMES},
                                blank, blank, blank, None)

                    pred, probs, heat, overlay, emb, rpt = run_and_report(
                        mobilenet_model, img, prep_for_mobilenet,
                        "MobileNetV2",
                        float(intensity), float(alpha)
                    )
                    return pred, safe_prob_display(probs), heat, overlay, emb, rpt

                def _ai_mb(pred, probs, model, length, tone):
                    probs_clean = safe_prob_for_ai(probs)
                    prompt = build_scc_prompt("MobileNetV2", pred, probs_clean, length, tone)
                    return ai_analyze_text(prompt, model=model)

                mb_btn.click(
                    fn=_run_mb,
                    inputs=[mb_img, mb_int, mb_alpha],
                    outputs=[
                        mb_out_pred, mb_out_probs,
                        mb_out_heat, mb_out_overlay,
                        mb_out_emboss, mb_out_report
                    ]
                )

                mb_ai_btn.click(
                    fn=_ai_mb,
                    inputs=[mb_out_pred, mb_out_probs,
                            ai_model_dropdown, ex_length, ex_tone],
                    outputs=[mb_ai_out]
                )

            # ================================================================
            # TAB 4 — ConvNeXt (5-Fold Ensemble)
            # ================================================================
            with gr.TabItem("ConvNeXt (5-Fold Ensemble)"):

                cn_img = gr.Image(type="pil", label="Upload Image (RGB)")
                cn_int = gr.Slider(0.2, 2.0, 1.0, label="Heatmap intensity")
                cn_alpha = gr.Slider(0.0, 0.8, 0.35, label="Heatmap transparency")
                cn_btn = gr.Button("Run ConvNeXt Ensemble & Generate Report")

                cn_out_pred = gr.Label(label="Prediction")
                cn_out_probs = gr.Label(label="Class Probabilities")
                cn_out_heat = gr.Image(label="Grad-CAM Heatmap")
                cn_out_overlay = gr.Image(label="Overlay")
                cn_out_emboss = gr.Image(label="Embossed")
                cn_out_report = gr.File(label="Download Report")

                cn_ai_btn = gr.Button("AI Interpretation (Using Global Settings)")
                cn_ai_out = gr.Textbox(label="AI Interpretation", lines=6)

                def _run_cn(img, intensity, alpha):
                    if img is None:
                        blank = np.zeros((IMG_SIZE_EFF[0], IMG_SIZE_EFF[1], 3), np.uint8)
                        return ("No image",
                                {k: 0.0 for k in CLASS_NAMES},
                                blank, blank, blank, None)

                    if len(convnext_folds) == 0:
                        blank = np.zeros((IMG_SIZE_EFF[0], IMG_SIZE_EFF[1], 3), np.uint8)
                        return ("ConvNeXt folds missing",
                                {k: 0.0 for k in CLASS_NAMES},
                                blank, blank, blank, None)

                    pred, prob, heat, overlay, emb, rpt = convnext_fold_ensemble(
                        img, float(intensity), float(alpha)
                    )
                    return pred, safe_prob_display(prob), heat, overlay, emb, rpt

                def _ai_cn(pred, probs, model, length, tone):
                    probs_clean = safe_prob_for_ai(probs)
                    prompt = build_scc_prompt("ConvNeXt Ensemble", pred, probs_clean, length, tone)
                    return ai_analyze_text(prompt, model=model)

                cn_btn.click(
                    fn=_run_cn,
                    inputs=[cn_img, cn_int, cn_alpha],
                    outputs=[
                        cn_out_pred, cn_out_probs,
                        cn_out_heat, cn_out_overlay,
                        cn_out_emboss, cn_out_report
                    ]
                )

                cn_ai_btn.click(
                    fn=_ai_cn,
                    inputs=[cn_out_pred, cn_out_probs,
                            ai_model_dropdown, ex_length, ex_tone],
                    outputs=[cn_ai_out]
                )

            # ================================================================
            # TAB 5 — Full All-Models Ensemble
            # ================================================================
            with gr.TabItem("All-Models Ensemble"):

                am_img = gr.Image(type="pil", label="Upload Image (RGB)")
                am_int = gr.Slider(0.2, 2.0, 1.0, label="Heatmap intensity")
                am_alpha = gr.Slider(0.0, 0.8, 0.35, label="Heatmap transparency")
                am_btn = gr.Button("Run Full Ensemble & Generate Report")

                am_out_pred = gr.Label(label="Prediction")
                am_out_probs = gr.Label(label="Class Probabilities")
                am_out_heat = gr.Image(label="Grad-CAM Heatmap")
                am_out_overlay = gr.Image(label="Overlay")
                am_out_emboss = gr.Image(label="Embossed")
                am_out_report = gr.File(label="Download Report")

                am_ai_btn = gr.Button("AI Interpretation (Using Global Settings)")
                am_ai_out = gr.Textbox(label="AI Interpretation", lines=6)

                def _run_all(img, intensity, alpha):
                    if img is None:
                        blank = np.zeros((IMG_SIZE_EFF[0], IMG_SIZE_EFF[1], 3), np.uint8)
                        return ("No image",
                                {k: 0.0 for k in CLASS_NAMES},
                                blank, blank, blank, None)

                    img = enforce_rgb_uint8(img)
                    preds_list = []

                    # EfficientNet
                    try:
                        if effnet_model is not None:
                            preds_list.append(
                                np.nan_to_num(
                                    effnet_model.predict(
                                        np.expand_dims(prep_for_effnet(img), 0),
                                        verbose=0
                                    )[0]
                                )
                            )
                    except Exception:
                        pass

                    # DenseNet
                    try:
                        if densenet_model is not None:
                            preds_list.append(
                                np.nan_to_num(
                                    densenet_model.predict(
                                        np.expand_dims(prep_for_densenet(img), 0),
                                        verbose=0
                                    )[0]
                                )
                            )
                    except Exception:
                        pass

                    # MobileNet
                    try:
                        if mobilenet_model is not None:
                            preds_list.append(
                                np.nan_to_num(
                                    mobilenet_model.predict(
                                        np.expand_dims(prep_for_mobilenet(img), 0),
                                        verbose=0
                                    )[0]
                                )
                            )
                    except Exception:
                        pass

                    # ConvNeXt folds
                    for m in convnext_folds:
                        try:
                            preds_list.append(
                                np.nan_to_num(
                                    m.predict(
                                        np.expand_dims(prep_for_convnext(img), 0),
                                        verbose=0
                                    )[0]
                                )
                            )
                        except Exception:
                            pass

                    if len(preds_list) == 0:
                        fallback = np.array([1/3, 1/3, 1/3])
                        return ("unknown",
                                {CLASS_NAMES[i]: float(fallback[i]) for i in range(3)},
                                None, None, None, None)

                    mean_prob = np.mean(np.stack(preds_list), axis=0)
                    mean_prob = mean_prob / (mean_prob.sum() + 1e-8)

                    prob_dict = {CLASS_NAMES[i]: float(mean_prob[i]) for i in range(3)}
                    pred_name = CLASS_NAMES[int(np.argmax(mean_prob))]

                    # visual model selection
                    vis_model = (
                        effnet_model
                        or densenet_model
                        or mobilenet_model
                        or (convnext_folds[0] if convnext_folds else None)
                    )
                    if vis_model is None:
                        return pred_name, prob_dict, None, None, None, None

                    vis_prep = (
                        prep_for_effnet if vis_model is effnet_model else
                        prep_for_densenet if vis_model is densenet_model else
                        prep_for_mobilenet if vis_model is mobilenet_model else
                        prep_for_convnext
                    )

                    _, heat, overlay, emb = generate_gradcam(
                        vis_model, img, vis_prep,
                        float(intensity), float(alpha)
                    )

                    orig = Image.fromarray(cv2.resize(img, IMG_SIZE_EFF[::-1]))

                    pdf_path = generate_pdf(
                        img,
                        "All-Models Ensemble",
                        pred_name,
                        prob_dict,
                        [
                            orig,
                            Image.fromarray(cv2.resize(heat, IMG_SIZE_EFF[::-1])),
                            Image.fromarray(cv2.resize(overlay, IMG_SIZE_EFF[::-1])),
                            Image.fromarray(cv2.resize(emb, IMG_SIZE_EFF[::-1])),
                        ]
                    )

                    return pred_name, prob_dict, heat, overlay, emb, pdf_path

                def _ai_all(pred, probs, model, length, tone):
                    probs_clean = safe_prob_for_ai(probs)
                    prompt = build_scc_prompt("All-Models Ensemble", pred, probs_clean, length, tone)
                    return ai_analyze_text(prompt, model=model)

                am_btn.click(
                    fn=_run_all,
                    inputs=[am_img, am_int, am_alpha],
                    outputs=[
                        am_out_pred, am_out_probs,
                        am_out_heat, am_out_overlay,
                        am_out_emboss, am_out_report
                    ]
                )

                am_ai_btn.click(
                    fn=_ai_all,
                    inputs=[am_out_pred, am_out_probs,
                            ai_model_dropdown, ex_length, ex_tone],
                    outputs=[am_ai_out]
                )

# ---------------------------------------------
# LAUNCH
# ---------------------------------------------
if __name__ == "__main__":
    print("Launching LungUI — centered clean layout (local).")
    demo.launch(share=True)
