# File: predict.py

import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

# Đảm bảo import đúng mô hình từ thư mục src
from src.FDSNet import FDSNet
import transforms as T # Import transforms của chúng ta

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    # --- Cấu hình ---
    num_classes = 104
    # Sửa đường dẫn đến model bạn muốn dùng để dự đoán
    weights_path = "./save_weights/model_30 (1).pth" # Ví dụ: dùng model của epoch 9
    img_path = "./test2.jpg" # Sửa đường dẫn đến ảnh bạn muốn dự đoán
    palette_path = "./palette.json"
    # ----------------

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # --- Sửa lỗi 1: Tạo mô hình cho đúng ---
    model = FDSNet(num_classes=num_classes)

    # --- Sửa lỗi 2: Thêm weights_only=False khi tải checkpoint ---
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # --- Pipeline xử lý ảnh đầu vào ---
    # Phải giống với pipeline khi validation/evaluation
    data_transform = T.Compose([
        T.Resize([768, 768]), # Resize về đúng kích thước training
        T.GenerateLaplacian(),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    original_img = Image.open(img_path).convert('RGB')
    # Áp dụng transform, nó sẽ trả về (img, aux_img, target)
    # Chúng ta không có target ở đây, nên có thể cần sửa nhẹ
    # Để đơn giản, ta sẽ gọi trực tiếp các lớp transform
    img_tensor, aux_tensor, _ = data_transform(original_img, original_img) # Dùng ảnh gốc làm target tạm thời

    # Mở rộng chiều batch
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    aux_tensor = torch.unsqueeze(aux_tensor, dim=0)

    model.eval()
    with torch.no_grad():
        t_start = time_synchronized()
        # Truyền tuple vào mô hình
        output = model((img_tensor.to(device), aux_tensor.to(device)))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = Image.fromarray(prediction)
        mask.putpalette(pallette)
        mask.save("test_result.png")

if __name__ == '__main__':
    main()