# app.py
import uuid
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from flask import Flask, request, render_template_string, send_from_directory, url_for

# -----------------------------
# CONFIG (match your training!)
# -----------------------------
SAMPLE_RATE = 16000
DURATION_SEC = 3.0
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

# fixed frames to match model input
TARGET_FRAMES = 1 + (TARGET_SAMPLES // HOP_LENGTH)

CENTER = True
LABELS = ["ANGER", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD"]
DEVICE = "cpu"

MODEL_WEIGHTS_PATH = "cnn_bilstm_emotion.pt"
TEMPERATURE = 2.0  # fixed (not shown in UI)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# -----------------------------
# Import your model architecture
# -----------------------------
from model import CNNBiLSTM  # must match weights

# -----------------------------
# Load model once at startup
# -----------------------------
model = CNNBiLSTM()
state = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# -----------------------------
# Audio preprocessing
# -----------------------------
def load_and_fix_length(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=None, mono=True)

    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode="constant")
    else:
        y = y[:TARGET_SAMPLES]

    return y

def wav_to_logmel(y: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
        center=CENTER
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.mean()) / (S_db.std() + 1e-6)
    return S_norm[np.newaxis, :, :].astype(np.float32)  # (1, n_mels, time)

def ensure_frames(feat: np.ndarray) -> np.ndarray:
    t = feat.shape[2]
    if t < TARGET_FRAMES:
        feat = np.pad(feat, ((0, 0), (0, 0), (0, TARGET_FRAMES - t)), mode="constant")
    elif t > TARGET_FRAMES:
        feat = feat[:, :, :TARGET_FRAMES]
    return feat

def predict_emotion(wav_path: str):
    y = load_and_fix_length(wav_path)
    feat = ensure_frames(wav_to_logmel(y))

    x = torch.from_numpy(feat).unsqueeze(0).to(DEVICE)  # (1, 1, 128, TARGET_FRAMES)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits / float(TEMPERATURE), dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    probs_percent = {LABELS[i]: float(probs[i].item() * 100.0) for i in range(len(LABELS))}
    return LABELS[pred_idx], probs_percent

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.lower().endswith(".wav")

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB

TML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Speech Emotion Recognition</title>
  <style>
    :root{
      --bg:#0b1220; --card:#0f1b33; --muted:#93a4c7; --text:#eaf0ff;
      --accent:#6ea8fe; --line:rgba(255,255,255,.08);
      --err:#fb7185;
    }
    *{box-sizing:border-box}
    body{
      margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      background:
        radial-gradient(1000px 600px at 10% 10%, rgba(110,168,254,.22), transparent),
        radial-gradient(800px 500px at 90% 30%, rgba(45,212,191,.16), transparent),
        var(--bg);
      color:var(--text);
      min-height:100vh;
    }

    .screen{
      min-height:100vh;
      display:flex;
      align-items:center;
      justify-content:center;
      padding: 28px 16px;
    }

    .wrap{ width:min(980px, 100%); }

    .header{
      display:flex;
      gap:14px;
      align-items:center;
      justify-content:center;
      text-align:center;
      margin-bottom: 80px;
    }

    .badge{
      width:46px; height:46px; border-radius:14px; background:rgba(110,168,254,.18);
      border:1px solid var(--line); display:grid; place-items:center;
      font-weight:800; color:var(--accent);
    }
    h1{margin:0; font-size:22px}
    .sub{margin:4px 0 0; color:var(--muted); font-size:14px}

    .grid{
      display:grid;
      grid-template-columns: 1.1fr .9fr;
      gap:14px;
    }
    @media (max-width: 860px){ .grid{grid-template-columns:1fr} }

    .card{
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
      border:1px solid var(--line);
      border-radius:18px;
      padding:16px;
      box-shadow: 0 20px 60px rgba(0,0,0,.35);
    }
    .card h2{margin:0 0 10px; font-size:16px}
    .muted{color:var(--muted); font-size:13px; line-height:1.4}
    .row{display:flex; gap:12px; flex-wrap:wrap; align-items:end}
    .field{flex:1; min-width:240px}
    label{display:block; font-size:13px; color:var(--muted); margin-bottom:6px}
    input[type="file"]{
      width:100%;
      padding:10px;
      border-radius:12px;
      border:1px solid var(--line);
      background: rgba(15,27,51,.55);
      color:var(--text);
      outline:none;
    }
    .btn{
      appearance:none;
      border:1px solid rgba(110,168,254,.45);
      background: rgba(110,168,254,.16);
      color: var(--text);
      padding:10px 14px;
      border-radius:12px;
      cursor:pointer;
      font-weight:600;
      transition:.15s transform, .15s background;
      white-space:nowrap;
    }
    .btn:hover{transform: translateY(-1px); background: rgba(110,168,254,.24)}
    .err{
      padding:10px 12px;
      border:1px solid rgba(251,113,133,.45);
      background: rgba(251,113,133,.12);
      color: var(--text);
      border-radius:12px;
      margin-bottom:12px;
    }
    .pill{
      display:inline-flex; align-items:center; gap:8px;
      padding:8px 10px;
      border-radius:999px;
      border:1px solid var(--line);
      background: rgba(15,27,51,.5);
      color: var(--muted);
      font-size:13px;
    }
    .pred{
      font-size:22px; font-weight:800; margin: 2px 0 12px;
      letter-spacing:.2px
    }
    .table{width:100%; border-collapse:collapse; margin-top:10px}
    .table th, .table td{
      padding:10px 8px; border-bottom:1px solid var(--line); font-size:13px;
    }
    .table th{color:var(--muted); font-weight:600; text-align:left}
    .bar{
      height:10px; border-radius:999px; background: rgba(255,255,255,.07);
      overflow:hidden; border:1px solid rgba(255,255,255,.06);
    }
    .bar > div{height:100%; background: linear-gradient(90deg, rgba(110,168,254,.9), rgba(45,212,191,.9));}
    .audio{margin-top:10px}
    audio{width:100%; margin-top:8px}
    .small{font-size:12px; color:var(--muted)}
    .hint{margin-top:10px}
  </style>
</head>
<body>
  <div class="screen">
    <div class="wrap">
      <div class="header">
        <div class="badge">SER</div>
        <div>
          <h1>Speech Emotion Recognition</h1>
          <div class="sub">Upload a WAV file → preview audio → Predict emotion</div>
        </div>
      </div>

      <div class="grid">
        <div class="card">
          <h2>Upload & Preview</h2>

          {% if error %}
            <div class="err"><b>Error:</b> {{ error }}</div>
          {% endif %}

          <form method="POST" action="/predict" enctype="multipart/form-data">
            <div class="row">
              <div class="field">
                <label>Audio file (.wav)</label>
                <input id="audioInput" type="file" name="audio" accept=".wav" required>
               
              </div>

              <div style="margin-left:auto;">
                <button class="btn" type="submit">Predict</button>
              </div>
            </div>

            <!-- Hidden until a file is selected -->
            <div class="audio" id="previewBlock" style="display:none;">
              <div class="pill">Preview</div>
              <audio id="previewPlayer" controls></audio>
              <div class="small">This preview plays directly from your browser selection (not uploaded yet).</div>
            </div>
          </form>

          {% if result %}
            <!-- Wrap uploaded section so we can hide it on new file select -->
            <div class="uploadedBlock">
              <hr style="border:0;border-top:1px solid var(--line);margin:14px 0;">
              <div class="pill">Uploaded: <b style="color:var(--text)">{{ result.original_name }}</b></div>
              <audio controls>
                <source src="{{ result.audio_url }}" type="audio/wav">
                Your browser does not support the audio element.
              </audio>
              <div class="small">This player streams the uploaded file from the server.</div>
            </div>
          {% endif %}
        </div>

        <div class="card">
          <h2>Result</h2>
          {% if result %}
            <div class="pred">{{ result.pred_label }}</div>

            <table class="table">
              <thead>
                <tr>
                  <th style="width:90px;">Emotion</th>
                  <th>Probability</th>
                  <th style="width:80px;">%</th>
                </tr>
              </thead>
              <tbody>
                {% for emo, val in result.probs %}
                  <tr>
                    <td><b>{{ emo }}</b></td>
                    <td><div class="bar"><div style="width: {{ val }}%"></div></div></td>
                    <td>{{ "%.2f"|format(val) }}</td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>

            <div class="small" style="margin-top:10px;">
              Note: probabilities are softmax outputs (relative preference), not calibrated real-world probabilities.
            </div>
          {% else %}
            <div class="muted">
              Upload a WAV file, preview it, then click <b>Predict</b> to get the emotion and probabilities.
            </div>
          {% endif %}
        </div>
      </div>
    </div>
  </div>

  <script>
    const input = document.getElementById("audioInput");
    const player = document.getElementById("previewPlayer");
    const previewBlock = document.getElementById("previewBlock");
    let objectUrl = null;

    input.addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];

      // Hide ANY previously uploaded audio blocks immediately
      document.querySelectorAll(".uploadedBlock").forEach(el => {
        el.style.display = "none";
      });

      // If user clears selection, hide preview again
      if (!file) {
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = null;
        player.removeAttribute("src");
        player.load();
        previewBlock.style.display = "none";
        return;
      }

      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(file);
      player.src = objectUrl;
      player.load();
      previewBlock.style.display = "block";
    });
  </script>
</body>
</html>
"""

@app.get("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)

@app.get("/")
def index():
    return render_template_string(TML, result=None, error=None)

@app.post("/predict")
def predict():
    if "audio" not in request.files:
        return render_template_string(TML, result=None, error="No file part in request.")

    f = request.files["audio"]
    if not f.filename:
        return render_template_string(TML, result=None, error="No file selected.")

    if not allowed_file(f.filename):
        return render_template_string(TML, result=None, error="Please upload a .wav file.")

    safe_name = f"{uuid.uuid4().hex}_{Path(f.filename).name}"
    save_path = UPLOAD_DIR / safe_name
    f.save(save_path)

    try:
        pred_label, probs_percent = predict_emotion(str(save_path))
        probs_sorted = sorted(probs_percent.items(), key=lambda kv: kv[1], reverse=True)

        result = {
            "pred_label": pred_label,
            "probs": probs_sorted,
            "original_name": Path(f.filename).name,
            "audio_url": url_for("uploaded_file", filename=safe_name),
        }
        return render_template_string(TML, result=result, error=None)

    except Exception as e:
        return render_template_string(TML, result=None, error=f"Inference failed: {e}")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
