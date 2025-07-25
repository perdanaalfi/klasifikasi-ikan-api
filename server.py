from flask import Flask, jsonify
from flask_cors import CORS
import joblib
from datetime import datetime
from collections import deque
import firebase_admin
from firebase_admin import credentials, db
import os
import json

app = Flask(__name__)
CORS(app)

# Inisialisasi Firebase jika belum aktif
if not firebase_admin._apps:
    firebase_json = os.environ.get("FIREBASE_CONFIG")
    if firebase_json:
        cred_dict = json.loads(firebase_json)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred, {
            "databaseURL": "https://klasifikasi-ikan-default-rtdb.asia-southeast1.firebasedatabase.app"
        })
        print("✅ Firebase berhasil diinisialisasi")
    else:
        print("⚠️ FIREBASE_CONFIG tidak ditemukan di environment variables")

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
    "waktu": None,
    "prediksi": []
}

# Aturan standar (rule-based) SNI
rules = {
    "lele":   {"suhu": [26, 32], "do": [3, 7], "ph": [6.5, 8]},
    "mas":    {"suhu": [20, 30], "do": [3, 7], "ph": [6.5, 8]},
    "nila":   {"suhu": [25, 30], "do": [3, 8], "ph": [6, 8.5]},
    "patin":  {"suhu": [26, 30], "do": [3, 7], "ph": [6.5, 8]},
    "gurame": {"suhu": [25, 28], "do": [5, 7], "ph": [6.5, 8]}
}

def save_to_firebase(data):
    """Simpan data ke Firebase Realtime Database"""
    try:
        if firebase_admin._apps:
            ref = db.reference('data_sensor')
            ref.push(data)
            print("✅ Data berhasil disimpan ke Firebase")
        else:
            print("⚠️ Firebase belum diinisialisasi")
    except Exception as e:
        print(f"❌ Error saat menyimpan ke Firebase: {e}")

def get_latest_from_firebase():
    """Ambil data terbaru dari Firebase"""
    try:
        if firebase_admin._apps:
            ref = db.reference('data_sensor')
            latest = ref.order_by_key().limit_to_last(1).get()
            if latest:
                return list(latest.values())[0]
        return None
    except Exception as e:
        print(f"❌ Error saat membaca dari Firebase: {e}")
        return None

def classify_fish(suhu, do, ph):
    """Klasifikasi ikan berdasarkan input dan rule + model"""
    hasil_model = []
    hasil_rule_dengan_skor = []

    # Prediksi dari model
    if model and (0 <= suhu <= 33) and (0 <= do <= 9) and (5 <= ph <= 9):
        try:
            pred = model.predict([[do, suhu, ph]])[0]
            hasil_model.append(pred.strip().lower())  # uniform lowercase
            print(f"🤖 Prediksi model: {pred}")
        except Exception as e:
            print("❌ Model error:", e)
    else:
        print("Input tidak wajar")

    # Rule-based dengan skor
    for ikan, batas in rules.items():
        skor = 0
        if batas["suhu"][0] <= suhu <= batas["suhu"][1]: skor += 1
        if batas["do"][0] <= do <= batas["do"][1]: skor += 1
        if batas["ph"][0] <= ph <= batas["ph"][1]: skor += 1
        if skor >= 2:
            hasil_rule_dengan_skor.append({
                "nama": ikan.strip().lower(),
                "skor": skor
            })
            print(f"📋 Rule cocok: {ikan} ({skor}/3)")

    # Urutkan berdasarkan skor tertinggi dulu
    hasil_rule_dengan_skor.sort(key=lambda x: x["skor"], reverse=True)
    
    # Gabungkan hasil model dengan rule (prioritas model dulu, lalu rule berdasarkan skor)
    hasil_final = []
    
    # Tambahkan hasil model dulu
    for model_fish in hasil_model:
        if model_fish not in [fish["nama"] for fish in hasil_rule_dengan_skor]:
            hasil_final.append(f"{model_fish.capitalize()} (Model)")
    
    # Tambahkan hasil rule berdasarkan skor
    for rule_fish in hasil_rule_dengan_skor:
        nama = rule_fish["nama"].capitalize()
        skor = rule_fish["skor"]
        
        # Cek apakah juga ada di hasil model
        if rule_fish["nama"] in hasil_model:
            hasil_final.append(f"{nama} (Model + Rule {skor}/3)")
        else:
            hasil_final.append(f"{nama} (Rule {skor}/3)")
    
    return hasil_final


@app.route("/")
def home():
    return "✅ Server Klasifikasi Ikan Aktif (Model + Rule-Based + Firebase)"

@app.route("/update/<suhu>/<do>/<ph>")
def update_data(suhu, do, ph):
    try:
        suhu = float(suhu)
        do = float(do)
        ph = float(ph)
    except:
        return jsonify({"status": "error", "message": "Data tidak valid"}), 400

    # Tambahkan ke history
    sensor_data = {
        "suhu": suhu,
        "do": do,
        "ph": ph,
        "waktu": datetime.now()
    }
    sensor_history.append(sensor_data)

    # Klasifikasi ikan berdasarkan data terbaru
    prediksi = classify_fish(suhu, do, ph)
    
    # Update data terakhir dengan prediksi
    last_data.update({
        "suhu": suhu,
        "do": do,
        "ph": ph,
        "waktu": datetime.now().strftime("%Y-%m-%d"),
        "prediksi": prediksi
    })

    # Simpan ke Firebase
    try:
        ref = db.reference('data_sensor')
        ref.push({
            "suhu": suhu,
            "do": do,
            "ph": ph,
            "prediksi": prediksi,
            "timestamp": datetime.now().strftime("%Y-%m-%d")
        })
    except Exception as firebase_error:
        print("⚠️ Gagal menyimpan ke Firebase:", firebase_error)

    print(f"📊 Data terbaru: Suhu={suhu}°C, DO={do}mg/L, pH={ph}")
    print(f"🐟 Prediksi: {prediksi}")

    return jsonify({"status": "ok", "prediksi": prediksi})

@app.route("/last-prediction")
def last_prediction():
    """Endpoint yang dipanggil website untuk mendapatkan data terbaru beserta prediksi"""
    return jsonify(last_data)

@app.route("/classify")
def classify():
    """Endpoint untuk klasifikasi berdasarkan rata-rata data history"""
    if not sensor_history:
        return jsonify({"prediksi": [], "message": "Belum ada data sensor"})

    # Hitung rata-rata
    avg_suhu = sum(d["suhu"] for d in sensor_history) / len(sensor_history)
    avg_do = sum(d["do"] for d in sensor_history) / len(sensor_history)
    avg_ph = sum(d["ph"] for d in sensor_history) / len(sensor_history)

    # Klasifikasi berdasarkan rata-rata
    hasil = classify_fish(avg_suhu, avg_do, avg_ph)

    return jsonify({
        "rata_rata": {
            "suhu": round(avg_suhu, 2),
            "do": round(avg_do, 2),
            "ph": round(avg_ph, 2)
        },
        "prediksi": hasil,
        "jumlah_data": len(sensor_history)
    })



@app.route("/firebase-data")
def get_firebase_data():
    """Endpoint untuk mengambil data dari Firebase"""
    try:
        if firebase_admin._apps:
            ref = db.reference('data_sensor')
            data = ref.order_by_key().limit_to_last(10).get()
            if data:
                return jsonify({"status": "ok", "data": data})
            else:
                return jsonify({"status": "ok", "data": {}, "message": "Tidak ada data"})
        else:
            return jsonify({"status": "error", "message": "Firebase belum diinisialisasi"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
