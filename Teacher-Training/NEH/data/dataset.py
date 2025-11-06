"""Dataset class for Retinal OCT images."""

import os
from torch.utils.data import Dataset
from PIL import Image


class RetinalOCTDataset(Dataset):
    """Dataset class with optional TTA support."""

    def __init__(self, dataframe, data_root, transform=None, use_tta=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.use_tta = use_tta
        self.class_to_idx = {'normal': 0, 'drusen': 1, 'cnv': 2}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        relative_path = str(self.dataframe.iloc[idx]['Directory'])
        img_path = os.path.join(self.data_root, relative_path)
        
        if not os.path.exists(img_path):
            corrected_path = relative_path.replace("NOrmal", "Normal")
            img_path = os.path.join(self.data_root, corrected_path)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        label_str = str(self.dataframe.iloc[idx]['Label']).lower()
        label = self.class_to_idx.get(label_str, 0)

        if self.use_tta and self.transform:
            # Return multiple augmented versions
            images = self.transform(image)
            return images, label
        elif self.transform:
            image = self.transform(image)

        return image, label

