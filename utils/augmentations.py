import random
import numpy as np


class Albumentations:
    def __init__(self, size=640):
        self.transform = None
        try:
            import albumentations as A
            T = [A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                 A.Blur(p=0.01),
                 A.MedianBlur(p=0.01),
                 A.ToGray(p=0.01),
                 A.CLAHE(p=0.01),
                 A.RandomBrightnessContrast(p=0.0),
                 A.RandomGamma(p=0.0),
                 A.ImageCompression(quality_lower=75, p=0.0)]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        except ImportError:  # package not installed, skip
            pass

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels
