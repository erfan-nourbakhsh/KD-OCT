"""Dataset for OCT images."""

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class MedOCTDataset(Dataset):
    """Enhanced dataset for OCT images with robust loading."""
    
    def __init__(
        self,
        dataframe,
        data_root: str,
        transform=None,
        cache_images: bool = False
    ):
        self.df = dataframe.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {}
        
        self.class_to_idx = {
            "normal": 0,
            "drusen": 1,
            "cnv": 2
        }
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int):
        img_rel = self.df.iloc[idx]["Directory"]
        relative_path = str(img_rel)
        
        # Check cache first
        if self.cache_images and idx in self.image_cache:
            image = self.image_cache[idx]
        else:
            img_path = os.path.join(self.data_root, relative_path)
            
            # Handle path corrections
            if not os.path.exists(img_path):
                corrected_path = relative_path.replace("NOrmal", "Normal")
                img_path = os.path.join(self.data_root, corrected_path)
            
            try:
                image = Image.open(img_path).convert("RGB")
                if self.cache_images:
                    self.image_cache[idx] = image
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                image = Image.new("RGB", (448, 448), (0, 0, 0))
        
        # Get label
        label_str = str(self.df.iloc[idx]["Label"]).lower()
        label = self.class_to_idx.get(label_str, 0)
        
        # Apply transforms
        if self.transform is not None:
            image_t = self.transform(image)
        else:
            image_t = T.ToTensor()(image)
        
        return image_t, label

