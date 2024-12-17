import numpy as np
from PIL import Image

def easy(image: np.ndarray, masks: np.ndarray):
    image_copy = image.copy()
    
    if image_copy.ndim == 3:
        image_copy[masks != 0] = [255, 255, 255]
    else:
        image_copy[masks != 0] = 255
    
    return Image.fromarray(image_copy)
