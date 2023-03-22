
from pathlib import Path
from PIL import Image
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from retinaface.utils import vis_annotations
from torch.utils.data import WeightedRandomSampler
from retinaface.pre_trained_models import get_model


def get_weighted_sampler(dataset: Dataset):
    """Returns sampler for Dataloader with label wheights"""
    spoof = 0
    live = 0
    for img, label in dataset:
        if label == 0:
            live += 1
        elif label == 1:
            spoof += 1
    sample_weights =[1/live, 1/spoof]

    print(live, spoof)

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)

    return sampler


def save_scaled_cropped_face(csv_path, image_dir):
    """Returns cropped images using bboxes from csv"""
    image_to_box = {}
    csv = pd.read_csv(csv_path)
    filename = os.path.basename(csv_path).split('.')[0]
    csv['image'] = csv['image'].str.replace(f'celeba/{filename}/', '')
    for _, row in csv.iterrows():
        image_name = row['image']
        label = list(row['box'][1:-1].split(', '))
        label = [int(x) for x in label]
        image_to_box[image_name] = label

    for img in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img)
        image = np.array(Image.open(img_path).convert('RGB'))
        print(image_to_box[img][0])
        image = image[image_to_box[img][1]:image_to_box[img][3], image_to_box[img][0]:image_to_box[img][2]]
        image = Image.fromarray(image)
        image.save(str(Path(image_dir, f'{img}')))


def visualize_retina_sample(image_path: Path):
    """Displays image with bbox returned from retinaface"""
    image = np.array(Image.open(image_path).convert('RGB'))
    model = get_model("resnet50_2020-07-20", max_size=2048)
    model.eval()
    annotation = model.predict_jsons(image, confidence_threshold=0.95, nms_threshold=0.1)
    plt.imshow(vis_annotations(image, annotation))
    plt.show()
