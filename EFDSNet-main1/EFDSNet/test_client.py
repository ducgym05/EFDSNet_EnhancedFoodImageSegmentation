# test_client.py
import requests
import json
import base64
from PIL import Image
import io

# --- THAY ĐỔI ĐỊA CHỈ NÀY ---
# Nếu chạy trên cùng máy, dùng: http://127.0.0.1:5000/predict
# Nếu chạy từ máy khác (hoặc điện thoại), 
# hãy tìm địa chỉ IP LAN của máy chủ (ví dụ: 192.168.1.10)
# và dùng: http://192.168.1.10:5000/predict
API_URL = "http://127.0.0.1:5000/predict"

# --- DỮ LIỆU ĐỂ TEST ---
IMAGE_PATH = "test2.jpg"  # Đảm bảo file này tồn tại
DISH_WEIGHT_G = float(input("Vui lòng nhập tổng trọng lượng đĩa ăn: "))     # Giả sử bạn cân đĩa được 450g

def test_api():
    print(f"Đang gửi request đến: {API_URL}")
    
    # Dữ liệu gửi đi (multipart/form-data)
    files = {
        'image': (IMAGE_PATH, open(IMAGE_PATH, 'rb'), 'image/jpeg')
    }
    data = {
        'weight': str(DISH_WEIGHT_G)
    }
    
    try:
        # Gửi request POST
        response = requests.post(API_URL, files=files, data=data)
        
        # In mã trạng thái
        print(f"Server response code: {response.status_code}")
        
        # Lấy kết quả JSON
        response_json = response.json()
        
        if response.status_code == 200:
            print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
            print(f"Tổng Calo: {response_json['total_calories_kcal']} kcal")
            print(f"Tổng Trọng lượng: {response_json['total_weight_g']} g")
            print("\nCác món ăn chi tiết:")
            print("-" * 50)
            print(f"{'Tên':<20} {'% Trọng lượng':<15} {'Trọng lượng(g)':<12} {'Calo(kcal)':<10}")
            print("-" * 50)
            
            for item in response_json['detected_items']:
                print(f"{item['name']:<20} {item['weight_ratio_percent']:<15.2f}% {item['weight_g']:<12.1f} {item['calories_kcal']:<10.0f}")
            
            # Giải mã ảnh và lưu lại
            img_base64 = response_json['segmentation_image_base64']
            img_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_bytes))
            
            output_filename = "api_test_result.png"
            img.save(output_filename)
            print(f"\nĐã lưu ảnh segmentation vào: {output_filename}")
            
        else:
            print("\n--- CÓ LỖI XẢY RA ---")
            print(json.dumps(response_json, indent=2))

    except requests.exceptions.ConnectionError as e:
        print(f"\nLỖI: Không thể kết nối đến server.")
        print("Bạn đã chạy 'python app.py' chưa?")
        print(f"Chi tiết: {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")

if __name__ == "__main__":
    test_api()