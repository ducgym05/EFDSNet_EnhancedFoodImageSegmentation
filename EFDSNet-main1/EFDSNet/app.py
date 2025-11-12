# app.py
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Import các hàm từ file logic của chúng ta
from predict_logic import initialize_globals, run_prediction

# ---- CẤU HINH SERVER ----
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__) 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ---- KHỞI TẠO MODEL (Chạy 1 lần duy nhất) ----
# (THAY ĐỔI) Tải thêm mô hình Depth Anything
print("Initializing models... (FDSNet and Depth Anything)")
model, device, palette_dict, palette, cat_names, data_transform, depth_processor, depth_model = initialize_globals()

# ---- CÁC HÀM TIỆN ÍCH ----
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---- ĐỊNH NGHĨA API ENDPOINT ----

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_image():
    # === 1. Kiểm tra request đầu vào ===
    if 'image' not in request.files:
        return jsonify({"error": "Could not find the 'image' part in the request."}), 400
    
    file = request.files['image']
        
    if file.filename == '':
        return jsonify({"error": "No file was selected."}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type (only .png, .jpg, .jpeg are supported)"}), 400

    # === 2. Lấy dữ liệu ===
    try:
        image_bytes = file.read()
    except Exception as e:
        return jsonify({"error": f"Failed to read data: {e}"}), 500

    # === 3. Chạy dự đoán ===
    
    # (THAY ĐỔI) Cập nhật hàm gọi run_prediction
    response_data, error = run_prediction(
        image_bytes=image_bytes,
        model=model,
        device=device,
        palette_dict=palette_dict,
        palette=palette,
        cat_names=cat_names,
        data_transform=data_transform,
        depth_processor=depth_processor, 
        depth_model=depth_model          
    )
    
    if error:
        return jsonify({"error": f"Server error: {error}"}), 500

    # === 4. Trả về kết quả ===
    print("Run successfully. Returning JSON.")
    return jsonify(response_data)

# ---- CHẠY SERVER ----
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)