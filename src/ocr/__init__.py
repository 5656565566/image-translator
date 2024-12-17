from .easy_ocr import EasyOCR
from .manga_ocr import MangaOCR

class OCR:
    def __init__(self, name: str, languages: str, **kwargs):
        self.name = name
        
        if self.name == "Easyocr":
            self.easyOCR = EasyOCR(languages, **kwargs)
        
        elif self.name == "MangaOCR":
            self.mangaOCR = MangaOCR(**kwargs)

    def __call__(self, bounding_boxes: list, crop_images: list) -> list[str]:
        
        texts = []
        bbox = []
        
        for bounding_boxe, crop_image in zip(bounding_boxes, crop_images):
            
            if self.name == "Easyocr":
                text = self.easyOCR(crop_image)
            elif self.name == "MangaOCR":
                text = self.mangaOCR.ocr_manga(crop_image)
            
            if text != "":
                texts.append(text)
                bbox.append(bounding_boxe)
                
        return texts, bbox
        
        