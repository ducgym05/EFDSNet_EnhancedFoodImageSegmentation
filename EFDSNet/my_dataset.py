import os
import torch.utils.data as data
from PIL import Image
import torch
import numpy as np

class VOCSegmentation(data.Dataset):
    def __init__(self, food_root, transforms=None, txt_name: str = "train.txt", m=None):
        super(VOCSegmentation, self).__init__()
        root = os.path.join(food_root, "FoodSeg103")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'Images', 'img_dir', m)
        mask_dir = os.path.join(root, 'Images', 'ann_dir', m)
        txt_path = os.path.join(root, "ImageSets", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip().replace('.jpg', '') for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            # transforms giờ trả về 3 giá trị
            img, aux, target = self.transforms(img, target)
            return (img, aux), target # Nhóm 2 ảnh vào một tuple
        
        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        # batch là một list các tuple [((img1, aux1), target1), ((img2, aux2), target2), ...]
        images_tuple, targets = list(zip(*batch))
        
        # Tách các tuple ảnh
        original_images, laplacian_images = list(zip(*images_tuple))

        # Gộp batch cho từng loại ảnh và target
        batched_imgs = cat_list(original_images, fill_value=0)
        batched_aux = cat_list(laplacian_images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        
        # Trả về một tuple ảnh và một tensor target
        return (batched_imgs, batched_aux), batched_targets

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs