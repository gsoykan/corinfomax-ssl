from typing import Optional, Any

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from custom_datasets.pytorch_utils import UnNormalize, visualize_tensor
from custom_datasets.transform_utils3 import PretrainTransform


class SSLComicsCropsDataset(Dataset):
    def __init__(self,
                 prefiltered_csv_folder_dir: Optional[str] = None,
                 transform: Optional[Any] = None,
                 item_type: str = 'body'):
        self.prefiltered_csv_folder_dir = prefiltered_csv_folder_dir
        self.item_type = item_type
        self.transform = transform
        self.dataset = self.load_dataset()

    def load_dataset(self):
        return pd.read_csv(self.prefiltered_csv_folder_dir)['img_path'].tolist()

    def __getitem__(self, index):
        source_raw_item = self.dataset[index]
        source_img = Image.open(source_raw_item)
        source_and_prime = self.transform(source_img)
        return source_and_prime

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    transform = PretrainTransform('comics_crops_bodies')
    unorm = UnNormalize(mean=[0.387, 0.480, 0.531],
                        std=[0.192, 0.225, 0.249], )
    source_raw_item = '/home/gsoykan20/Desktop/self_development/amazing-mysteries-of-gutter-demystified/data/comics_crops/0/0_1/bodies/0.jpg'
    source_img = Image.open(source_raw_item)
    e, e_prime = transform(source_img)
    visualize_tensor(e, unorm)
    visualize_tensor(e_prime, unorm)
    print(e)
    print(e, e_prime)
