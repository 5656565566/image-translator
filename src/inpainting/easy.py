import numpy as np

from PIL import Image

def easy(image: np.ndarray, masks: np.ndarray):
    
    output_image = (image[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    combined_mask = np.logical_or.reduce(masks)
    output_image[combined_mask] = [255, 255, 255]
    
    return Image.fromarray(output_image)