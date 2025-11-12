# predict_with_calories_and_color.py
import os
import json
import torch
import numpy as np
from PIL import Image
from collections import Counter
import re

from src.FDSNet import FDSNet
import transforms as T

# ====================== CẤU HÌNH ======================
WEIGHTS_PATH = "./save_weights/model_120.pth"
IMG_PATH = "./test4.jpg"
PALETTE_PATH = "./palette.json"
CATEGORY_PATH = "./category_id.txt"
OUTPUT_MASK = "test_result_calories.png"
OUTPUT_LABELS = "detected_foods_calories.txt"
MIN_PIXEL_RATIO = 0.005 
# =================== ==================================

# BẢNG CALO (kcal/100g) - mở rộng từ USDA
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


# <--- THAY ĐỔI: Thêm Bảng Hệ số Mật độ Tương đối (Ước lượng)
# Đây là giá trị ước lượng, 1.0 = trung bình
# > 1.0 = nặng/đặc (thịt, phô mai, các loại hạt)
# < 1.0 = nhẹ/xốp (rau, bánh mì, bỏng ngô)
RELATIVE_DENSITY_FACTOR = {
    # Mặc định là 1.0, chỉ liệt kê các trường hợp đặc biệt
    # Nhẹ/Xốp
    6: 0.2,  # popcorn (rất nhẹ)
    58: 0.5, # bread
    80: 0.3, # lettuce (rất nhẹ)
    102: 0.3,# salad (rất nhẹ)
    75: 0.4, # seaweed
    87: 0.6, # broccoli (cồng kềnh)
    72: 0.6, # cauliflower
    # Nặng/Đặc
    9: 1.8,  # cheese butter (rất đặc)
    46: 1.5, # steak
    47: 1.4, # pork
    51: 1.4, # lamb
    17: 1.6, # almond
    19: 1.6, # cashew
    22: 1.6, # walnut
    23: 1.6, # peanut
    66: 1.3, # rice (đặc hơn nước)
    64: 1.2, # pasta
    70: 1.2, # potato
    24: 1.1, # egg
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

def main():
    for p in [WEIGHTS_PATH, IMG_PATH, PALETTE_PATH, CATEGORY_PATH]:
        assert os.path.exists(p), f"Không tìm thấy: {p}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

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

    img = Image.open(IMG_PATH).convert('RGB')
    dummy_mask = Image.new('L', img.size, 255)
    img_tensor, aux_tensor, _ = data_transform(img, dummy_mask)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    aux_tensor = aux_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model((img_tensor, aux_tensor))
        pred = output['out'].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

    mask = Image.fromarray(pred)
    mask.putpalette(palette)
    mask.save(OUTPUT_MASK)
    print(f"Đã lưu segmentation map: {OUTPUT_MASK}")

    # <--- THAY ĐỔI: LOGIC TÍNH TOÁN MỚI
    
    # Bước 1: Thu thập pixel và "pixel đã điều chỉnh" (adjusted pixels)
    unique_labels = np.unique(pred)
    detected_foods_data = []
    h, w = pred.shape
    total_pixels = h * w
    total_adjusted_pixels = 0.0

    for label in unique_labels:
        if label == 0:  # background
            continue
        
        name = cat_names.get(label, f"unknown_{label}")
        pixel_count = np.sum(pred == label)
        pixel_ratio = pixel_count / total_pixels
        
        # Lấy hệ số mật độ, mặc định là 1.0
        density_factor = RELATIVE_DENSITY_FACTOR.get(label, 1.0)
        adjusted_pixels = pixel_count * density_factor
        total_adjusted_pixels += adjusted_pixels
        
        # Tra cứu màu
        color_rgb = palette_dict.get(str(label), [0, 0, 0])
        color_str = f"R:{color_rgb[0]}, G:{color_rgb[1]}, B:{color_rgb[2]}"

        detected_foods_data.append({
            'label': label,
            'name': name,
            'pixel_ratio': pixel_ratio, # Tỷ lệ diện tích (để tham khảo)
            'adjusted_pixels': adjusted_pixels,
            'color_str': color_str
        })

    # Bước 2: Hỏi người dùng tổng trọng lượng
    total_dish_weight_g = 0.0
    total_dish_weight_g = float(input("Vui lòng nhập tổng trọng lượng của đĩa ăn: "))
    # while total_dish_weight_g <= 0:
    #     try:
    #         # <--- THAY ĐỔI: Hỏi người dùng
    #         user_input = input("\n>>> Vui lòng nhập TỔNG TRỌNG LƯỢNG của đĩa ăn (gram): ")
    #         total_dish_weight_g = float(user_input)
    #         if total_dish_weight_g <= 0:
    #             print("Trọng lượng phải là số dương.")
    #     except ValueError:
    #         print("Vui lòng nhập một con số (ví dụ: 450.5)")

    # Bước 3: Tính toán trọng lượng và calo dựa trên "tỷ lệ trọng lượng" (weight_ratio)
    total_calories = 0.0
    final_results = []

    if total_adjusted_pixels == 0:
        print("Không phát hiện thấy pixel thực phẩm nào (total_adjusted_pixels = 0).")
    else:
        for data in detected_foods_data:
            # Tỷ lệ trọng lượng (ước tính)
            weight_ratio = data['adjusted_pixels'] / total_adjusted_pixels
            weight_g = weight_ratio * total_dish_weight_g
            
            kcal_per_100g = CALORIES_PER_100G.get(data['label'], 100)  # default 100
            calories = (weight_g / 100) * kcal_per_100g
            total_calories += calories
            
            data['weight_ratio'] = weight_ratio
            data['weight_g'] = weight_g
            data['calories'] = calories
            
            # Chỉ thêm vào kết quả cuối cùng nếu vượt ngưỡng diện tích
            if data['pixel_ratio'] > MIN_PIXEL_RATIO:
                final_results.append(data)

    # Sắp xếp theo trọng lượng (weight_g) giảm dần
    final_results = sorted(final_results, key=lambda x: x['weight_g'], reverse=True)

    # IN KẾT QUẢ
    print("\n" + "="*85)
    print(f"MÓN ĂN & ƯỚC LƯỢNG CALO (Tổng trọng lượng nhập vào: {total_dish_weight_g:.1f}g)")
    print("(Ước tính trọng lượng dựa trên tỷ lệ diện tích ĐÃ ĐIỀU CHỈNH theo mật độ)")
    print("="*85)
    print(f"{'STT':<3} {'MÓN ĂN':<20} {'DIỆN TÍCH':<10} {'TRỌNG LƯỢNG':<13} {'TRỌNG LƯỢNG(g)':<12} {'CALO':<8} {'MÀU (RGB)':<20}")
    print("-" * 85)
    
    for i, data in enumerate(final_results):
        print(f"{i+1:<3} {data['name']:<20} {data['pixel_ratio']*100:9.2f}% {data['weight_ratio']*100:12.2f}%  {data['weight_g']:10.1f}   {data['calories']:6.0f}   {data['color_str']:<20}")
    
    print("-" * 85)
    print(f"TỔNG CALO ƯỚC LƯỢNG (cho {total_dish_weight_g:.1f}g): {total_calories:.0f} kcal")

    # LƯU VÀO FILE
    with open(OUTPUT_LABELS, 'w', encoding='utf-8') as f:
        f.write("Danh sách món ăn & calo ước lượng (đã điều chỉnh mật độ):\n")
        f.write("="*90 + "\n")
        f.write(f"Tổng trọng lượng đĩa (do người dùng nhập): {total_dish_weight_g:.1f}g\n")
        f.write(f"Tổng calo (ước tính): {total_calories:.0f} kcal\n\n")
        f.write(f"{'STT':<3} {'MÓN':<20} {'DIỆN TÍCH':<10} {'TRỌNG LƯỢNG':<13} {'g':<12} {'kcal':<8} {'MÀU (RGB)':<20}\n")
        f.write("-" * 90 + "\n")
        
        for i, data in enumerate(final_results):
            f.write(f"{i+1:<3} {data['name']:<20} {data['pixel_ratio']*100:9.2f}% {data['weight_ratio']*100:12.2f}%  {data['weight_g']:10.1f}   {data['calories']:6.0f}   {data['color_str']:<20}\n")
        
        f.write("-" * 90 + "\n")
        f.write(f"TỔNG: {total_calories:.0f} kcal\n")
    print(f"\nĐã lưu chi tiết vào: {OUTPUT_LABELS}")

if __name__ == '__main__':
    main()