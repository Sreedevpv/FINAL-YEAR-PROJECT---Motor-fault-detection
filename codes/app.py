from flask import Flask, jsonify, render_template
import numpy as np
import sounddevice as sd
import librosa
import pickle
import threading
import time
import collections

app = Flask(__name__)

# ===== LOAD MODEL =====
# Load the sklearn Pipeline (contains scaler + RF inside)
with open('fault_detector_rf.pkl', 'rb') as f:
    model = pickle.load(f)

# ===== CONFIG =====
MIC_SR    = 44100   # mic sample rate
MODEL_SR  = 22050   # model expects 22050
N_MFCC    = 40      # must match training
DURATION  = 1.0     # seconds per prediction window

# Smoothing: keep last N predictions, show majority vote
SMOOTH_WINDOW = 5
prediction_history = collections.deque(maxlen=SMOOTH_WINDOW)

# Shared state
latest_audio      = np.zeros(int(MODEL_SR * DURATION))
latest_prediction = "Starting..."
latest_confidence = 0.0
latest_rms        = 0.0
is_running        = True

# ===== FEATURE EXTRACTION =====
# Must exactly match rfModle.ipynb extract_features()
def extract_features(y, sr=MODEL_SR):
    feats = []

    # ── MFCCs mean + std (40 × 2 = 80)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    feats.extend(np.mean(mfcc, axis=1))
    feats.extend(np.std(mfcc,  axis=1))

    # ── Delta MFCCs mean + std (40 × 2 = 80)
    delta_mfcc = librosa.feature.delta(mfcc)
    feats.extend(np.mean(delta_mfcc, axis=1))
    feats.extend(np.std(delta_mfcc,  axis=1))

    # ── Spectral features (5 × 2 = 10)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw       = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr           = librosa.feature.zero_crossing_rate(y)
    rms           = librosa.feature.rms(y=y)

    for feat in [spec_centroid, spec_bw, spec_rolloff, zcr, rms]:
        feats.append(np.mean(feat))
        feats.append(np.std(feat))

    # ── Spectral contrast mean + std (7 × 2 = 14)
    feats.extend(np.mean(spec_contrast, axis=1))
    feats.extend(np.std(spec_contrast,  axis=1))

    # ── Chroma mean + std (12 × 2 = 24)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.extend(np.mean(chroma, axis=1))
    feats.extend(np.std(chroma,  axis=1))

    # ── Tonnetz mean + std (6 × 2 = 12)
    try:
        tonnetz = librosa.feature.tonnetz(
            y=librosa.effects.harmonic(y), sr=sr
        )
        feats.extend(np.mean(tonnetz, axis=1))
        feats.extend(np.std(tonnetz,  axis=1))
    except Exception:
        feats.extend([0.0] * 12)

    return np.array(feats, dtype=np.float32)


# ===== AUDIO LOOP =====
def audio_loop():
    global latest_audio, latest_prediction, latest_confidence, latest_rms

    while is_running:
        try:
            # Record
            audio = sd.rec(
                int(MIC_SR * DURATION),
                samplerate=MIC_SR,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            audio = audio.flatten()

            # Resample to model SR
            audio_resampled = librosa.resample(
                audio, orig_sr=MIC_SR, target_sr=MODEL_SR
            )
            latest_audio = audio_resampled.copy()

            # RMS level (for waveform amplitude display)
            latest_rms = float(np.sqrt(np.mean(audio_resampled ** 2)))

            # Skip near-silent clips
            if latest_rms < 0.001:
                latest_prediction = "Listening..."
                latest_confidence = 0.0
                continue

            # Extract features & predict
            feats = extract_features(audio_resampled, sr=MODEL_SR)
            feats_2d = feats.reshape(1, -1)

            # model is a Pipeline — already has scaler inside
            raw_pred  = model.predict(feats_2d)[0]           # 0 or 1
            prob      = model.predict_proba(feats_2d)[0]     # [p_normal, p_fault]

            # Smooth with majority vote
            prediction_history.append(int(raw_pred))
            smoothed = int(round(
                sum(prediction_history) / len(prediction_history)
            ))

            label = "FAULT" if smoothed == 1 else "NORMAL"
            confidence = float(prob[smoothed])

            latest_prediction = label
            latest_confidence = round(confidence * 100, 1)

        except Exception as e:
            print(f"[audio_loop error] {e}")
            latest_prediction = "Error"
            latest_confidence = 0.0
            time.sleep(0.5)


# ===== ROUTES =====
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data")
def data():
    # Downsample waveform for browser (send 200 points max)
    waveform = latest_audio
    if len(waveform) > 200:
        step = len(waveform) // 200
        waveform = waveform[::step][:200]

    return jsonify({
        "audio":      waveform.tolist(),
        "prediction": latest_prediction,
        "confidence": latest_confidence,
        "rms":        round(latest_rms * 1000, 2)
    })


# ===== START THREAD =====
threading.Thread(target=audio_loop, daemon=True).start()

if __name__ == "__main__":
    print("🔧 Machine Fault Detector running → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
