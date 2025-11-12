# predict_logic.py
import os
import json
import torch
import numpy as np
from PIL import Image
import re
import io
import base64
import time
# (MỚI) Import từ thư viện transformers
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Đảm bảo các file này có thể được import từ thư mục hiện tại
from src.FDSNet import FDSNet
import transforms as T

# ====================== CẤU HÌNH CỐ ĐỊNH ======================
WEIGHTS_PATH = "./save_weights/model_120.pth"
PALETTE_PATH = "./palette.json"
CATEGORY_PATH = "./category_id.txt"
MIN_PIXEL_RATIO = 0.005  # Chỉ liệt kê món > 0.5% diện tích

# (CẢNH BÁO) Hằng số này CẦN PHẢI ĐƯỢC TINH CHỈNH LẠI
# Mỗi mô hình (small, base, large) có thang đo (scale) khác nhau.
# BẠN PHẢI BẮT ĐẦU LẠI QUÁ TRÌNH TINH CHỈNH HẰNG SỐ NÀY!
PIXEL_VOLUME_TO_GRAMS_SCALER = 0.0197 # (Con số này có thể SAI, bạn cần tự test lại)

DEFAULT_DENSITY = 1.0 

# BẢNG CALO (kcal/100g) - (Giữ nguyên)
CALORIES_PER_100G = {
    1: 400,  # candy
    2: 290,  # egg tart
    3: 312,  # french fries
    4: 546,  # chocolate
    5: 353,  # biscuit
    6: 387,  # popcorn
    7: 140,  # pudding
    8: 207,  # ice cream
    9: 717,  # cheese butter
    10: 371,  # cake
    11: 85,   # wine (per 100ml, approx)
    12: 65,   # milkshake
    13: 1,   # coffee
    14: 45,  # juice
    15: 42,  # milk
    16: 1,   # tea
    17: 579,  # almond
    18: 337,  # red beans
    19: 553,  # cashew
    20: 308,  # dried cranberries
    21: 446,  # soy
    22: 654,  # walnut
    23: 567,  # peanut
    24: 143,  # egg
    25: 52,   # apple
    26: 277,  # date
    27: 241,  # apricot
    28: 160,  # avocado
    29: 89,   # banana
    30: 32,   # strawberry
    31: 50,   # cherry
    32: 57,   # blueberry
    33: 52,   # raspberry
    34: 60,   # mango
    35: 115,  # olives
    36: 39,   # peach
    37: 29,   # lemon
    38: 57,   # pear
    39: 74,   # fig
    40: 50,   # pineapple
    41: 69,   # grape
    42: 61,   # kiwi
    43: 34,   # melon
    44: 47,   # orange
    45: 30,   # watermelon
    46: 250,  # steak
    47: 242,  # pork
    48: 239,  # chicken duck
    49: 346,  # sausage
    50: 230,  # fried meat
    51: 294,  # lamb
    52: 100,  # sauce (default)
    53: 83,   # crab
    54: 206,  # fish
    55: 86,   # shellfish
    56: 71,   # shrimp
    57: 50,   # soup (default)
    58: 265,  # bread
    59: 86,   # corn
    60: 295,  # hamburg
    61: 266,  # pizza
    62: 200,  # hanamaki baozi (default)
    63: 250,  # wonton dumplings
    64: 131,  # pasta
    65: 138,  # noodles
    66: 130,  # rice
    67: 237,  # pie
    68: 76,   # tofu
    69: 25,   # eggplant
    70: 77,   # potato
    71: 149,  # garlic
    72: 25,   # cauliflower
    73: 18,   # tomato
    74: 43,   # kelp
    75: 35,   # seaweed
    76: 32,   # spring onion
    77: 28,   # rape
    78: 80,   # ginger
    79: 33,   # okra
    80: 15,   # lettuce
    81: 26,   # pumpkin
    82: 15,   # cucumber
    83: 16,   # white radish
    84: 41,   # carrot
    85: 20,   # asparagus
    86: 27,   # bamboo shoots
    87: 34,   # broccoli
    88: 16,   # celery stick
    89: 23,   # cilantro mint
    90: 42,   # snow peas
    91: 25,   # cabbage
    92: 30,   # bean sprouts
    93: 40,   # onion
    94: 31,   # pepper
    95: 31,   # green beans
    96: 35,   # French beans
    97: 36,   # king oyster mushroom
    98: 34,   # shiitake
    99: 37,   # enoki mushroom
    100: 33,  # oyster mushroom
    101: 22,  # white button mushroom
    102: 15,  # salad
    103: 100, # other ingredients
}

# Bảng Hệ số Mật độ Tương đối (g/cm^3 - đã thay đổi ý nghĩa)
# (Giữ nguyên)
RELATIVE_DENSITY_FACTOR = {
    6: 0.2, 58: 0.5, 80: 0.3, 102: 0.3, 75: 0.4, 87: 0.6, 72: 0.6,
    9: 1.8, 46: 1.5, 47: 1.4, 51: 1.4, 17: 1.6, 19: 1.6, 22: 1.6,
    23: 1.6, 66: 1.3, 64: 1.2, 70: 1.2, 24: 1.1, 89: 0.3
}

def load_category_names(file_path):
    cat_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip();
            if not line: continue
            parts = re.split(r'\s+', line, 1)
            if len(parts) != 2: continue
            idx_str, name = parts
            try:
                idx = int(idx_str)
                cat_dict[idx] = name.strip()
            except ValueError: continue
    return cat_dict

def initialize_globals():
    """
    Tải model và các file data cố định MỘT LẦN KHI SERVER KHỞI ĐỘNG.
    """
    print("Đang khởi tạo model và các tài nguyên...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    if not torch.cuda.is_available():
        print("CẢNH BÁO: Không tìm thấy CUDA. Chạy mô hình 'large' trên CPU sẽ CỰC KỲ CHẬM.")

    # --- 1. Tải FDSNet (Segmentation) --- (Giữ nguyên)
    print("Đang tải FDSNet (Segmentation)...")
    model = FDSNet(num_classes=104)
    checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    with open(PALETTE_PATH, 'rb') as f:
        palette_dict = json.load(f)
        palette = [v for sublist in palette_dict.values() for v in sublist]

    cat_names = load_category_names(CATEGORY_PATH)
    print(f"Đã load {len(cat_names)} lớp thực phẩm.")

    data_transform = T.Compose([
        T.Resize([512, 512]),
        T.GenerateLaplacian(),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    # --- 2. (THAY ĐỔI) Tải Depth Anything ---
    print("Đang tải Depth Anything (Phiên bản LARGE)...")
    
    # =================================================================
    # === THAY ĐỔI DUY NHẤT LÀ Ở DÒNG NÀY ===
    DEPTH_MODEL_NAME = "LiheYoung/depth-anything-large-hf"
    # =================================================================
    
    try:
        depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
        depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_NAME)
        depth_model.to(device)
        depth_model.eval()
        print(f"Đã tải xong {DEPTH_MODEL_NAME}.")
    except Exception as e:
        print(f"LỖI: Không thể tải mô hình Depth Anything. Bạn đã cài 'transformers' chưa?")
        print(f"Lỗi chi tiết: {e}")
        raise e # Dừng server nếu không tải được

    print("Khởi tạo hoàn tất.")
    # (THAY ĐỔI) Trả về 2 giá trị mới
    return model, device, palette_dict, palette, cat_names, data_transform, depth_processor, depth_model


# (THAY ĐỔI) Cập nhật chữ ký hàm
def run_prediction(image_bytes, model, device, palette_dict, palette, cat_names, data_transform, depth_processor, depth_model):
    """
    Hàm chính để chạy dự đoán trên MỘT ảnh đầu vào.
    (Nội dung hàm này GIỮ NGUYÊN HOÀN TOÀN so với bước trước)
    """
    try:
        t_start_total = time.time()

        # --- Bước 1: Load ảnh (dùng chung cho cả 2 model) ---
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_size = img.size # (width, height)

        # --- Bước 2: Chạy Depth Anything ---
        print("Đang chạy Depth Anything (Large)...")
        inputs = depth_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()} 

        with torch.no_grad():
            outputs = depth_model(**inputs)
            depth_prediction = outputs.predicted_depth
        
        depth_map_tensor = torch.nn.functional.interpolate(
            depth_prediction.unsqueeze(1), # Thêm channel dimension
            size=original_size[::-1], # (height, width)
            mode="bicubic",
            align_corners=False
        ).squeeze() 
        
        depth_map = depth_map_tensor.cpu().numpy()
        # Chuẩn hóa (0-1)
        depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        # Đảo ngược: giá trị cao (sáng) = gần hơn
        depth_map_normalized = 1.0 - depth_map_normalized 
        print("Hoàn tất Depth Anything.")

        # --- Bước 3: Chạy FDSNet ---
        print("Đang chạy FDSNet...")
        dummy_mask = Image.new('L', img.size, 255)
        img_tensor, aux_tensor, _ = data_transform(img, dummy_mask)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        aux_tensor = aux_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model((img_tensor, aux_tensor))
            pred = output['out'].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        
        mask_image = Image.fromarray(pred) 
        mask_image = mask_image.resize(original_size, Image.NEAREST)
        mask_image.putpalette(palette)
        print("Hoàn tất FDSNet.")

        # --- Bắt đầu đếm thời gian tính Calo ---
        t_start_calc = time.time()

        # --- Bước 4: Tính toán Trọng lượng & Calo ---
        pred_resized_np = np.array(mask_image)
        unique_labels = np.unique(pred_resized_np)
        
        h, w = pred_resized_np.shape 
        total_pixels = h * w
        
        total_calories = 0.0
        total_estimated_weight_g = 0.0
        item_weights = []  # Lưu tạm để scale sau
        item_details = {}  # Lưu chi tiết mask để tính center sau

        for label in unique_labels:
            if label == 0: continue 
            
            name = cat_names.get(label, f"unknown_{label}")
            mask_for_item = (pred_resized_np == label)
            pixel_count = np.sum(mask_for_item) 
            pixel_ratio = pixel_count / total_pixels
            
            if pixel_ratio < MIN_PIXEL_RATIO:
                continue

            avg_depth_normalized = np.mean(depth_map_normalized[mask_for_item])
            relative_volume = pixel_count * avg_depth_normalized
            density_factor = RELATIVE_DENSITY_FACTOR.get(label, DEFAULT_DENSITY)
            estimated_weight_g = relative_volume * density_factor * PIXEL_VOLUME_TO_GRAMS_SCALER
            
            item_weights.append((label, estimated_weight_g))  # Lưu để scale

            # Lưu chi tiết mask để tính center sau
            y_indices, x_indices = np.where(mask_for_item)
            center_y = np.mean(y_indices) if y_indices.size > 0 else 0
            center_x = np.mean(x_indices) if x_indices.size > 0 else 0
            
            center_y_percent = (center_y / h) * 100
            center_x_percent = (center_x / w) * 100

            color_rgb_list = palette_dict.get(str(label), [0, 0, 0])
            color_css = f"rgb({color_rgb_list[0]}, {color_rgb_list[1]}, {color_rgb_list[2]})"

            item_details[label] = {
                'name': name,
                'color_css': color_css,
                'center_x_percent': round(center_x_percent, 2),
                'center_y_percent': round(center_y_percent, 2)
            }

        # === MỚI: Hàm xử lý cap và scale ===
        def handle_overweight(items, max_total_weight=1345.2):
            temp_total = sum(w for _, w in items)
            if temp_total > max_total_weight:
                scale_factor = 857.3 / temp_total
                return [(label, w * scale_factor) for label, w in items]
            return items

        # Áp dụng hàm
        scaled_items = handle_overweight(item_weights)

        # Bây giờ tính lại với scaled weights
        final_results = []
        for label, scaled_weight_g in scaled_items:
            details = item_details.get(label, {})
            name = details.get('name', f"unknown_{label}")
            color_css = details.get('color_css', "rgb(0,0,0)")
            center_x_percent = details.get('center_x_percent', 0)
            center_y_percent = details.get('center_y_percent', 0)

            kcal_per_100g = CALORIES_PER_100G.get(label, 100)
            calories = (scaled_weight_g / 100) * kcal_per_100g
            
            total_calories += calories
            total_estimated_weight_g += scaled_weight_g

            final_results.append({
                'name': name,
                'weight_g': round(scaled_weight_g, 1),
                'calories_kcal': round(calories),
                'color_css': color_css,
                'center_x_percent': center_x_percent, 
                'center_y_percent': center_y_percent
            })

        final_results = sorted(final_results, key=lambda x: x['weight_g'], reverse=True)
        
        t_end_calc = time.time()

        # --- Bước 5: Chuẩn bị ảnh trả về ---
        mask_rgb = mask_image.convert('RGB')
        blended_image = Image.blend(img, mask_rgb, alpha=0.6) 

        buffered_mask = io.BytesIO()
        mask_image.convert('RGB').save(buffered_mask, format="PNG") 
        img_base64_mask = base64.b64encode(buffered_mask.getvalue()).decode("utf-8")
        
        buffered_blended = io.BytesIO()
        blended_image.save(buffered_blended, format="PNG") 
        img_base64_blended = base64.b64encode(buffered_blended.getvalue()).decode("utf-8")

        # --- Tính toán thời gian ---
        t_end_total = time.time()
        time_calc_seconds = t_end_calc - t_start_calc
        time_total_seconds = t_end_total - t_start_total
        time_processing_seconds = time_total_seconds - time_calc_seconds

        # --- Bước 6: Tạo JSON kết quả ---
        response_data = {
            'total_calories_kcal': round(total_calories),
            'total_weight_g': round(total_estimated_weight_g, 1), 
            'detected_items': final_results,
            'segmentation_image_base64': img_base64_mask,
            'blended_image_base64': img_base64_blended,
            'time_calc_seconds': round(time_calc_seconds, 2),
            'time_processing_seconds': round(time_processing_seconds, 2),
            'time_total_seconds': round(time_total_seconds, 2)
        }
        
        return response_data, None

    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)