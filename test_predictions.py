import numpy as np
import torch
import os
from PIL import Image
import torch.nn as nn
from const import Config
from pathlib import Path
from model import EfficientNetB0
from dataset import val_transform



def predict_test(test_dir: Path, dir_path: Path):
    """
    Return printed predictions as following: Image: image_name.jpg, Label: 0- live, 1- spoof
    """

    model = EfficientNetB0()
    model.to(Config.DEVICE)
    model.load_state_dict(torch.load(Path(dir_path, 'model.best.pth')))
    model.eval()
    sigmoid = nn.Sigmoid()

    with torch.no_grad():
        for i, img in enumerate(os.listdir(test_dir)):
            test_img = np.array(
                Image.open(Path(test_dir, f'{img}')).convert('RGB')).astype(np.float32) / 255.0
            transformed = val_transform(image=test_img)
            test_img = transformed["image"]
            test_img = test_img.unsqueeze(0)
            output = model(test_img.to(Config.DEVICE))
            output = sigmoid(output).cpu().detach().numpy()
            print(f'Image: {img}, Label: {int(output>0.5)}')


if __name__ == '__main__':

    predict_test(Config.TEST_DIR, Config.DIR_PATH)
