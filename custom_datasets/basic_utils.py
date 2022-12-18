import itertools
import json
import os
from enum import Enum
from typing import List, Optional, Callable

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def search_files(extension='.ttf',
                 folder='H:\\',
                 filename_condition: Optional[Callable[[str], bool]] = None,
                 limit: Optional[int] = None,
                 enable_tqdm: bool = False):
    if limit:
        files = []
        for r, d, f in (tqdm(os.walk(folder), desc='walking in folders...') if enable_tqdm else os.walk(folder)):
            if limit is not None and len(files) >= limit:
                break
            for file in f:
                if file.endswith(extension):
                    filename = r + "/" + file
                    if filename_condition is not None:
                        if filename_condition(filename):
                            files.append(filename)
                    else:
                        files.append(filename)
        return files
    else:
        # TODO: @gsoykan - add filename condition to alt-search files 2 ... it causes bugs
        return alternative_search_files_2(extension, folder)


def scandir_walk(top):
    for entry in os.scandir(top):
        if entry.is_dir(follow_symlinks=False):
            yield from scandir_walk(entry.path)
        else:
            yield entry.path


# this is way faster...
def alternative_search_files_2(extension=".ttf", folder="H:\\") -> List[str]:
    return [
        file for file in tqdm(scandir_walk(folder), desc="walking in folder..")
        if file.endswith(extension)
    ]


def read_or_get_image(img,
                      read_rgb: bool = False):
    img_str = ""
    if not isinstance(img, (np.ndarray, str)):
        raise AssertionError('Images must be strings or numpy arrays')

    if isinstance(img, str):
        img_str = img
        img = cv2.imread(img)

    if img is None:
        raise AssertionError('Image could not be read: ' + img_str)

    if read_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def read_or_get_image_masked(img,
                             read_rgb: bool = True,
                             masks: List = [],
                             mask_fill_value=0,
                             return_pil_image: bool = True):
    read_image = read_or_get_image(img, read_rgb=read_rgb)
    for bb in masks:
        read_image[int(bb[1]): int(bb[3]), int(bb[0]):int(bb[2])] = mask_fill_value
    return Image.fromarray(read_image) if return_pil_image else read_image


# source: https://stackoverflow.com/a/42728126/8265079 | https://stackoverflow.com/questions/42727586/nest-level-of-a-list
def nest_level(obj):
    # Not a list? So the nest level will always be 0:
    if type(obj) != list:
        return 0
    # Now we're dealing only with list objects:
    max_level = 0
    for item in obj:
        # Getting recursively the level for each item in the list,
        # then updating the max found level:
        max_level = max(max_level, nest_level(item))
    # Adding 1, because 'obj' is a list (here is the recursion magic):
    return max_level + 1


def read_ad_pages(ad_page_path: str = '../../data/ad_pages_original.txt'):
    with open(ad_page_path) as f:
        lines = f.readlines()
    ad_pages = []
    for line in lines:
        comic_no, page_no = line.strip().split('---')
        ad_pages.append((int(comic_no), int(page_no)))
    return ad_pages


def flatten_list(l: List):
    return list(itertools.chain(*l))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def inverse_normalize(tensor,
                      mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225),
                      in_place: bool = True):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    if not in_place:
        tensor = tensor.clone()
    tensor.mul_(std).add_(mean)
    return tensor


class OCRFileKey(str, Enum):
    COMIC_NO = 'comic_no'
    PAGE_NO = 'page_no'
    PANEL_NO = 'panel_no'
    TEXTBOX_NO = 'textbox_no'
    DIALOG_OR_NARRATION = 'dialog_or_narration'
    TEXT = 'text'
    x1 = 'x1'
    y1 = 'y1'
    x2 = 'x2'
    y2 = 'y2'


if __name__ == '__main__':
    pass
