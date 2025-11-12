import torch
from torch import nn
import train_utils.distributed_utils as utils

def criterion(inputs, target, class_labels):
    """
    Tính loss cho cả segmentation và auxiliary classification.
    - Segmentation: CrossEntropyLoss với ignore_index=255.
    - Classification: BCEWithLogitsLoss cho multi-label (104 lớp).
    Trọng số loss: 1.0 cho segmentation, 0.1 cho classification (có thể điều chỉnh).
    """
    seg_loss = nn.functional.cross_entropy(inputs['out'], target, ignore_index=255)
    cls_loss = nn.BCEWithLogitsLoss()(inputs['aux'], class_labels.float())
    total_loss = seg_loss + 0.3 * cls_loss  # Cân bằng loss
    return total_loss

def evaluate(model, data_loader, device, num_classes):
    """
    Đánh giá mô hình trên tập validation.
    - Tính confusion matrix cho segmentation.
    - Tính accuracy cho auxiliary classification.
    """
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    total_correct_cls = 0
    total_samples_cls = 0

    with torch.no_grad():
        for image_tuple, target, class_labels in metric_logger.log_every(data_loader, 100, header):
            original_img, laplacian_img = image_tuple
            
            original_img = original_img.to(device)
            laplacian_img = laplacian_img.to(device)
            target = target.to(device)
            class_labels = class_labels.to(device)

            output = model((original_img, laplacian_img))
            seg_output = output['out']
            aux_output = output['aux']
            
            # Cập nhật confusion matrix cho segmentation
            confmat.update(target, seg_output)

            # Tính accuracy cho auxiliary classification
            aux_pred = (aux_output > 0.5).float()  # Ngưỡng 0.5 cho binary classification
            total_correct_cls += (aux_pred == class_labels).sum().item()
            total_samples_cls += class_labels.numel()

        confmat.reduce_from_all_processes()
        
        # Tính accuracy classification
        acc_cls = total_correct_cls / total_samples_cls if total_samples_cls > 0 else 0.0
        print(f"Auxiliary Classification Accuracy: {acc_cls:.4f}")
    
    return confmat

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    """
    Huấn luyện mô hình trong một epoch.
    - Hỗ trợ dữ liệu với nhãn classification phụ (class_labels).
    - Sử dụng mixed precision nếu scaler được cung cấp.
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image_tuple, target, class_labels in metric_logger.log_every(data_loader, print_freq, header):
        original_img, laplacian_img = image_tuple
        original_img, laplacian_img = original_img.to(device), laplacian_img.to(device)
        target = target.to(device)
        class_labels = class_labels.to(device)

        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            output = model((original_img, laplacian_img))
            loss = criterion(output, target, class_labels)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr

def create_lr_scheduler(optimizer, num_step: int, epochs: int, warmup=True, warmup_epochs=1, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if not warmup:
        warmup_epochs = 0

    def f(x):
        if warmup and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)