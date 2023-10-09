import re
import torch
import json
import random
import numpy as np
from PIL import Image
from pathlib import Path
from loguru import logger
from torch.utils.data import Dataset
from transformers import AutoProcessor
from transformers.models.clip import CLIPModel, CLIPProcessor, CLIPConfig


class FlickrDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset_path,
                 data_type: str = "train",
                 processor_path: str = "../model_name_or_path",
                 en_dataset: bool = False,
                 shuffle: bool = True):
        if not data_type or not dataset_path:
            raise ValueError("dataset path or data type is None")
        # init
        self.data_type = data_type
        self.dataset_path = dataset_path
        self.en_dataset = en_dataset  # whether using english dataset
        self.shuffle = shuffle

        # loading annotation lines
        self.annotation_lines = self.load_flickr()
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        txt_id = 0
        for img_id, ann in enumerate(self.annotation_lines):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for index, text in enumerate(ann['caption']):
                self.text.append(self.pre_caption(text, 77))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

        # loading image processor
        self.clip_processor = CLIPProcessor.from_pretrained(processor_path)

    def __getitem__(self, index):
        image_path = Path(self.dataset_path) / (self.image[index].replace("\\", "/"))
        image_pil = Image.open(image_path)
        text = self.text[np.random.choice(self.img2txt[index])]
        inputs = self.clip_processor(text=[text], images=image_pil, return_tensors="pt", padding="max_length", max_length=77)
        return inputs

    def __len__(self):
        return len(self.annotation_lines)

    @staticmethod
    def pre_caption(caption, max_words):
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption

    def load_flickr(self):
        annotation_lines = None
        if self.data_type.lower() == "train":
            if not self.en_dataset:
                annotation_lines = json.load(
                    open(Path(self.dataset_path) / "cn_train.json", mode='r', encoding='utf-8'))
            else:
                annotation_lines = json.load(
                    open(Path(self.dataset_path) / "en_train.json", mode='r', encoding='utf-8'))
        elif self.data_type.lower() == "test":
            if not self.en_dataset:
                annotation_lines = json.load(open(Path(self.dataset_path) / "cn_val.json", mode='r', encoding='utf-8'))
            else:
                annotation_lines = json.load(open(Path(self.dataset_path) / "en_val.json", mode='r', encoding='utf-8'))
        return annotation_lines


# if __name__ == "__main__":
#     s = "flickr8k-images\\3393035454_2d2370ffd4.jpg"
#     print(s.replace("\\", "/"))
#     for batch in FlickrDataset("../data", "test"):
#         print(batch['input_ids'].shape)
#         print(batch['pixel_values'].shape)
