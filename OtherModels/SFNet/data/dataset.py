"""OCT Dataset for SF-Net training."""

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class OCTDatasetSFNet(Dataset):
    """OCT Dataset for SF-Net training."""
    
    def __init__(self, dataframe, data_root: str, image_size: int = 224, is_training: bool = True):
        self.df = dataframe.reset_index(drop=True)
        self.data_root = data_root
        self.class_to_idx = {"normal": 0, "drusen": 1, "cnv": 2}
        
        if is_training:
            self.transform = T.Compose([
                T.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int):
        img_rel = self.df.iloc[idx]["Directory"]
        img_path = os.path.join(self.data_root, str(img_rel))
        
        # Handle path corrections
        if not os.path.exists(img_path):
            corrected = str(img_rel).replace("NOrmal", "Normal")
            img_path = os.path.join(self.data_root, corrected)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        label_str = str(self.df.iloc[idx]["Label"]).lower()
        label = self.class_to_idx.get(label_str, 0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

