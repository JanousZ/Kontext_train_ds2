import pandas as pd
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class MotionEditDataset(Dataset):
    def __init__(self, files_path):
        dfs = []
        for file_path in files_path:
            data = pd.read_parquet(file_path)
            dfs.append(data)
        self.data = pd.concat(dfs, ignore_index=True)

        self.resolutions = {}
        for i in range(len(self.data)):
            # 假设存储结构如你所写，我们只读 header 获取 size 以加快速度
            row = self.data.iloc[i]
            with Image.open(BytesIO(row["input_image"]["bytes"])) as img:
                if img.size not in self.resolutions:
                    self.resolutions[img.size] = 0

    def __len__(self):
        return len(self.data)

    def image_preprocess(self, image):
        img = np.array(image)
        img = img.astype(np.float32) / 127.5 - 1.0  # -> [-1,1]
        img = torch.from_numpy(img).permute(2, 0, 1)  # -> [C,H,W]
        img = img.to(dtype=torch.float32)
        return img

    def __getitem__(self, index):
        row = self.data.iloc[index]

        raw_input_image = Image.open(BytesIO(row["input_image"]["bytes"]))
        raw_target_image = Image.open(BytesIO(row["target_image"]["bytes"]))
        input_image =  self.image_preprocess(raw_input_image)    
        target_image = self.image_preprocess(raw_target_image)           

        return {
            "prompt": row["prompt"],
            "input_image": input_image, #Image.open(BytesIO(row["input_image"]["bytes"])),
            "target_image": target_image, #Image.open(BytesIO(row["target_image"]["bytes"]))
        }

files_path = ['train-00000-of-00006.parquet',
              'train-00001-of-00006.parquet',
              'train-00002-of-00006.parquet',
              'train-00003-of-00006.parquet',
              'train-00004-of-00006.parquet',
              'train-00005-of-00006.parquet']
motionedit_dataset = MotionEditDataset(files_path)
# print(motionedit_dataset[0]["input_image"].shape)



