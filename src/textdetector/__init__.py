from .ctd import TextDetector as CTD
from .efficient_sam import EfficientSam

import numpy as np

from PIL import Image

models = {
    "ctd": CTD,
    "EfficientSam": EfficientSam
}

class Textdetector:
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        
        if name == "ctd":
            self.textDetector = CTD(**kwargs)
        
        elif name == "EfficientSam":
            self.efficientSam = EfficientSam(**kwargs)

    def __call__(self, image: Image):
        if self.name == "ctd":
            _, mask_refined, blk_list = self.textDetector.predict(image)
            
            bounding_boxes = []
            
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                width, height = x2 - x1, y2 - y1
                
                # 增加 20% 的空间
                expanded_x1 = x1 - width * 0.1
                expanded_y1 = y1 - height * 0.1
                expanded_x2 = x2 + width * 0.1
                expanded_y2 = y2 + height * 0.1
                
                bounding_boxes.append([expanded_x1, expanded_y1, expanded_x2, expanded_y2])
            
            return mask_refined, bounding_boxes
        
        elif self.name == "EfficientSam":
            
            bounding_boxes, masks = self.efficientSam.predict(image)
            
            combined_mask = np.logical_or.reduce(masks)
            
            return combined_mask, bounding_boxes