import os
import torch
import torch.nn as nn
from const import Config
from clearml import Task
from dataset import Celeba
import torch.optim as optim
import random
import numpy as np
from catalyst import dl
from model import EfficientNetB0
from torch.utils.data import random_split
from dataset import train_transform, val_transform
from custom_metrics import CustomAccuracyMetric, CustomPrecisionMetric, CustomRecallMetric

if __name__ == '__main__':

    with torch.no_grad():
        torch.cuda.empty_cache()


    def seed_everything(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


    seed_everything(42)

    task_name = 'repeat'
    task = Task.init(project_name='face-antispoofing', task_name=task_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = Config.DEVICE

    train_dataset = Celeba(Config.IMG_DIR_TRAIN, Config.CSV_TRAIN, transform=train_transform)
    val_dataset = Celeba(Config.IMG_DIR_VAL, Config.CSV_VAL, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = EfficientNetB0()
    model.to(Config.DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LR
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=1
    )
    runner = dl.SupervisedRunner(
        input_key="features", output_key="scores", target_key="target", loss_key="loss"
    )

    loaders = {
        "train": train_loader,
        "valid": val_loader,
    }

    callbacks = [
            dl.BatchMetricCallback(
                metric=CustomAccuracyMetric(),
                input_key="scores",
                target_key="target",
                log_on_batch=True),
            dl.BatchMetricCallback(
                metric=CustomPrecisionMetric(),
                input_key="scores",
                target_key="target",
                log_on_batch=True),
            dl.BatchMetricCallback(
                metric=CustomRecallMetric(),
                input_key="scores",
                target_key="target",
                log_on_batch=True)
        ]

    runner.train(
        model=model,
        criterion=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        num_epochs=Config.EPOCH,
        callbacks=callbacks,
        logdir=f"./logdir",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
    )
