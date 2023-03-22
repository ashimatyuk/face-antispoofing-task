from pathlib import Path
from PIL import Image
import os
import numpy as np
from const import Config
from retinaface.pre_trained_models import get_model
device = Config.DEVICE


def get_scaled_cropped_face(model, image_dir):
    """Saves cropped faces images with 0.5 bboxes expanded scale"""
    model.eval()
    for img in os.listdir(image_dir):
        print(img)
        img_path = os.path.join(image_dir, img)
        image = np.array(Image.open(img_path).convert('RGB'))
        # image_tensor = torch.from_numpy(image).float().to(device)
        # image_tensor = image_tensor.cpu().numpy()
        annotation = model.predict_jsons(image, confidence_threshold=0.95, nms_threshold=0.01)
        bbox = annotation[0]['bbox']
        if len(bbox) == 0:
            os.remove(img_path)
        else:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            bbox_new = [
                round(max(0, bbox[0] - width * 0.5)),
                round(max(0, bbox[1] - height * 0.5)),
                round(min(image.shape[1], bbox[2] + width * 0.5)),
                round(min(image.shape[0], bbox[3] + height * 0.5))
            ]
            if bbox_new[0] >= 0 and bbox_new[1] >= 0 and bbox_new[2] <= image.shape[1] and bbox_new[3] <= image.shape[0]:
                image = image[bbox_new[1]:bbox_new[3], bbox_new[0]:bbox_new[2]]
                image = Image.fromarray(image)
                image.save(str(Path(image_dir, f'{img}')))
            else:
                print(f'scaled bbox on img {img} is larger than input image')


if __name__ == '__main__':

    model = get_model("resnet50_2020-07-20", max_size=2048, device='cuda')
    get_scaled_cropped_face(model, Config.IMG_DIR_TRAIN)
    get_scaled_cropped_face(model, Config.IMG_DIR_VAL)


