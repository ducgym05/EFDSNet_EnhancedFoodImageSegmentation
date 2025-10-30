import torch
from torch import nn
import train_utils.distributed_utils as utils

def criterion(inputs, target):
    # inputs giờ là một dictionary, ta lấy key 'out'
    losses = nn.functional.cross_entropy(inputs['out'], target, ignore_index=255)
    return losses

# File: train_utils/train_and_eval.py

# File: train_utils/train_and_eval.py

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image_tuple, target in metric_logger.log_every(data_loader, 100, header):
            original_img, laplacian_img = image_tuple
            
            original_img = original_img.to(device)
            laplacian_img = laplacian_img.to(device)
            target = target.to(device)

            output = model((original_img, laplacian_img))
            output = output['out']
            
            # Dòng đã được sửa, không còn .flatten() hay .argmax(1)
            confmat.update(target, output)

        confmat.reduce_from_all_processes()
    return confmat

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        original_img, laplacian_img = image
        original_img, laplacian_img = original_img.to(device), laplacian_img.to(device)
        target = target.to(device)

        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            output = model((original_img, laplacian_img))
            loss = criterion(output, target)

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