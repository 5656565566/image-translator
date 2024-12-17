import numpy as np

from .lama import load_lama_mpe
from .easy import easy

from PIL import Image

class Inpainting:
    def __init__(self, name: str, **kwargs):
        self.name = name
        
        if name == "lama":
            self.lama = load_lama_mpe(**kwargs)

    def __call__(self, image: np.ndarray, masks: np.ndarray) -> Image:
        if self.name == "lama":
            
            predicted_img = self.lama.predict(image, masks)
            
            predicted_img = (predicted_img * 255).clip(0, 255).astype(np.uint8)
            
            return Image.fromarray(predicted_img)
        
        elif self.name == "easy":
            return easy(image, masks)