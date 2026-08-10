import os
import cv2
import base64
import json
import datetime
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO
from pathlib import Path
from PIL import Image as PILImage
import google.generativeai as genai
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///wood_defects.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

UPLOAD_FOLDER = 'static/uploads'
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# --- EXPERT KNOWLEDGE BASE ---
DEFECT_KNOWLEDGE = {
    "knot": {
        "desc": "عقدة طبيعية ناتجة عن نمو الأغصان.",
        "impact": "تؤدي لتغيير اتجاه الألياف مما قد يقلل من قوة الشد في هذه المنطقة."
    },
    "dead_knot": {
        "desc": "عقدة ميتة (سوداء) منفصلة عن أنسجة الخشب المحيطة.",
        "impact": "عالية الخطورة؛ قد تسقط وتترك فجوة، مما يضعف الهيكل الإنشائي ويشوه المظهر."
    },
    "knot_with_crack": {
        "desc": "عقدة مصابة بتشققات داخلية أو حولها.",
        "impact": "تؤدي لضعف شديد واحتمالية انكسار اللوح عند تعرضه لأحمال ميكانيكية."
    },
    "crack": {
        "desc": "تشقق طولي في ألياف الخشب نتيجة الجفاف أو الإجهاد.",
        "impact": "يقلل من متانة اللوح بشكل كبير ويسمح بنفاذ الرطوبة والحشرات للداخل."
    },
    "mold": {
        "desc": "نمو فطري على سطح الخشب.",
        "impact": "يؤدي لتلف الألياف بمرور الوقت وقد يسبب مشاكل صحية وتغير في اللون."
    }
}

# --- MODELS ---
print("🚀 [VERIFIED VERSION 2.1] Starting NexQA Expert Engine...")
model = YOLO('best.pt')
print("✅ YOLO Model Loaded Successfully.")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

class ProductRecord(db.Model):
    id = db.Column(db.String(50), primary_key=True)
    image_base64 = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    defect_type = db.Column(db.String(100), nullable=False)
    buyer = db.Column(db.String(100))
    shipping = db.Column(db.String(100))
    inspected_at = db.Column(db.String(50))

with app.app_context():
    db.create_all()

# --- ROUTES ---
@app.route('/')
def index():
    return "<h1>🚀 NexQA Expert Engine is Running</h1>"

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image"}), 400
    
    file = request.files['image']
    img_path = Path(UPLOAD_FOLDER) / f"mob_{file.filename}"
    file.save(img_path)
    
    results = model.predict(img_path, conf=0.15, verbose=False)
    
    yolo_detections = []
    expert_report = []
    
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            yolo_detections.append(label)
            
            # Get Expert Knowledge
            info = DEFECT_KNOWLEDGE.get(label.lower(), {
                "desc": f"تم اكتشاف {label} بواسطة نظام الرؤية.",
                "impact": "يؤثر على جودة المنتج النهائي."
            })
            
            expert_report.append({
                "name": label,
                "description": info["desc"],
                "impact": info["impact"],
                "confidence": conf
            })

    if not yolo_detections:
        final_status = "passed"
        expert_report = [{
            "name": "None",
            "description": "الخشب سليم تماماً ومطابق للمواصفات القياسية.",
            "impact": "لا يوجد أي تأثير سلبي؛ اللوح جاهز للاستخدام الإنشائي.",
            "confidence": 1.0
        }]
    else:
        final_status = "rejected"

    # Gemini Fallback (Optional)
    try:
        image_pil = PILImage.open(img_path)
        prompt = f"خبير جودة. الموديل وجد: {yolo_detections}. اشرح التأثير الفني باختصار JSON."
        gemini_resp = gemini_model.generate_content([prompt, image_pil], request_options={"timeout": 5})
        import re
        json_match = re.search(r'\{.*\}', gemini_resp.text, re.DOTALL)
        if json_match and yolo_detections:
             # If Gemini works, we can enrich the report
             pass 
    except Exception as e:
        print(f"⚠️ Gemini Skipped: {e}")

    # Plot Detections
    res = results[0]
    processed_img = res.plot()
    _, buffer = cv2.imencode('.jpg', processed_img)
    encoded_string = base64.b64encode(buffer).decode('utf-8')
    
    record_id = f"#{db.session.query(ProductRecord).count() + 1}"
    new_record = ProductRecord(
        id=record_id, image_base64=encoded_string, status=final_status,
        confidence=expert_report[0]['confidence'], defect_type=expert_report[0]['name'],
        buyer="NexQA Final Verified Station", shipping="Air Cargo",
        inspected_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    db.session.add(new_record)
    db.session.commit()

    return jsonify({
        "success": True, "status": final_status, "data": expert_report,
        "image_base64": encoded_string, "id": record_id,
        "buyer": new_record.buyer, "shipping": new_record.shipping
    })

@app.route('/api/history', methods=['GET'])
def api_history():
    records = ProductRecord.query.order_by(ProductRecord.inspected_at.desc()).all()
    return jsonify({"success": True, "history": [
        {"id": r.id, "status": r.status, "confidence": r.confidence, "defect_type": r.defect_type, 
         "buyer": r.buyer, "shipping": r.shipping, "inspected_at": r.inspected_at, "image_base64": r.image_base64} 
        for r in records
    ]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
