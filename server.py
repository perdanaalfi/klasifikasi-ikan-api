from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
from datetime import datetime

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load("model_ikan_rf.pkl")
except:
    model = None  # agar server tetap jalan meskipun belum ada model

# Data terbaru dari sensor
last_data = {
    "suhu": 0.0,
    "do": 0.0,
    "ph": 0.0,
    "prediksi": "-",
    "waktu": None
}

@app.route("/")
def home():
    return "✅ Server Klasifikasi Ikan Aktif"

@app.route("/last-prediction")
def last_prediction():
    return jsonify(last_data)

@app.route("/update/<suhu>/<do>/<ph>")
def update_data(suhu, do, ph):
    try:
        suhu = float(suhu)
        do = float(do)
        ph = float(ph)
    except ValueError:
        return jsonify({"status": "error", "message": "Format angka tidak valid"}), 400

    pred = "-"
    if model:
        pred = model.predict([[do, suhu, ph]])[0]

    last_data.update({
        "suhu": suhu,
        "do": do,
        "ph": ph,
        "prediksi": pred,
        "waktu": datetime.now().isoformat()
    })

    return jsonify({"status": "OK", "prediksi": pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
