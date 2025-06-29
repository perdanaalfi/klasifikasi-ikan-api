from flask import Flask, jsonify
from flask_cors import CORS
import joblib
from datetime import datetime
from collections import deque

app = Flask(__name__)
CORS(app)

# Load model Random Forest
try:
    model = joblib.load("model_ikan_rf.pkl")
    print("✅ Model berhasil dimuat.")
except Exception as e:
    print("❌ Gagal memuat model:", e)
    model = None

# Riwayat data sensor selama 1 menit (12 data, asumsikan 5 detik sekali)
sensor_history = deque(maxlen=12)

# Data terakhir
last_data = {
    "suhu": 0,
    "do": 0,
    "ph": 0,
    "waktu": None
}

# Aturan standar (rule-based) SNI
rules = {
    "lele":   {"suhu": [22, 33], "do": [2, 6], "ph": [6, 9]},
    "mas":    {"suhu": [20, 30], "do": [3, 8], "ph": [6.5, 9]},
    "nila":   {"suhu": [20, 33], "do": [3, 8], "ph": [6, 9]},
    "patin":  {"suhu": [24, 30], "do": [3, 7], "ph": [6, 8.5]},
    "gurame": {"suhu": [24, 30], "do": [3, 7], "ph": [6, 8.5]}
}

@app.route("/")
def home():
    return "✅ Server Klasifikasi Ikan Aktif (Model + Rule-Based)"

@app.route("/update/<suhu>/<do>/<ph>")
def update_data(suhu, do, ph):
    try:
        suhu = float(suhu)
        do = float(do)
        ph = float(ph)
    except:
        return jsonify({"status": "error", "message": "Data tidak valid"}), 400

    sensor_history.append({
        "suhu": suhu,
        "do": do,
        "ph": ph,
        "waktu": datetime.now()
    })

    last_data.update({
        "suhu": suhu,
        "do": do,
        "ph": ph,
        "waktu": datetime.now().isoformat()
    })

    return jsonify({"status": "ok"})

@app.route("/last-prediction")
def last_prediction():
    return jsonify(last_data)

@app.route("/classify")
def classify():
    if not sensor_history:
        return jsonify({"prediksi": [], "message": "Belum ada data sensor"})

    # Hitung rata-rata
    avg_suhu = sum(d["suhu"] for d in sensor_history) / len(sensor_history)
    avg_do = sum(d["do"] for d in sensor_history) / len(sensor_history)
    avg_ph = sum(d["ph"] for d in sensor_history) / len(sensor_history)

    hasil = []

    # Prediksi dari model (jika tersedia)
    if model:
        try:
            pred = model.predict([[avg_do, avg_suhu, avg_ph]])[0]
            hasil.append(pred)
        except Exception as e:
            print("❌ Gagal prediksi dari model:", e)

    # Tambahan dari rule-based
    for ikan, batas in rules.items():
        skor = 0
        if batas["suhu"][0] <= avg_suhu <= batas["suhu"][1]: skor += 1
        if batas["do"][0] <= avg_do <= batas["do"][1]: skor += 1
        if batas["ph"][0] <= avg_ph <= batas["ph"][1]: skor += 1

        if skor >= 2 and ikan not in hasil:
            hasil.append(ikan)

    return jsonify({
        "rata_rata": {
            "suhu": round(avg_suhu, 2),
            "do": round(avg_do, 2),
            "ph": round(avg_ph, 2)
        },
        "prediksi": hasil,
        "jumlah_data": len(sensor_history)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
