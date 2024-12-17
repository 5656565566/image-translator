from ocr import OCR
from inpainting import Inpainting
from rendering import TextEmbedder
from translators import Translator
from textdetector import Textdetector

from PIL import Image
from pathlib import Path

import numpy as np
import os

textDetector = Textdetector(name= "ctd")
ocr = OCR(name= "MangaOCR", languages="ja")
inpainting = Inpainting(name= "easy", device = "cuda")
textEmbedder = TextEmbedder(name= "easy")
translator = Translator(name= "SakuraTranslator", api_url= "http://192.168.100.113:8080/v1/chat/completions")

def process_images(input_folder, test_folder):
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                
                image_path = Path(root) / file
                
                print(f"正在找字 {image_path.parent.name} / {file}")
                
                image = Image.open(image_path)
                
                mask, bounding_boxes = textDetector(image)
                
                croped_images = []
                
                print(f"正在OCR {image_path.parent.name} / {file}")
                
                for bounding_boxe in bounding_boxes:
                    croped_images.append(image.crop(bounding_boxe))
                
                source_text, bbox = ocr(bounding_boxes, croped_images)
                
                if len(croped_images) != len(source_text):
                    print("OCR 结果与文本框的数量没有对应 !")
                
                texts = translator(source_text)
                
                if len(croped_images) != len(texts):
                    print("翻译结果与文本框的数量没有对应 !")
                
                print(f"正在翻译 {image_path.parent.name} / {file}")
                
                predicted_img = inpainting(np.array(image), mask)
                
                print(f"正在修复 {image_path.parent.name} / {file}")
                
                result_image = textEmbedder(texts, predicted_img, bbox)
                
                print(f"正在嵌字 {image_path.parent.name} / {file}")
                
                if not os.path.exists(test_folder / image_path.parent.name):
                    os.makedirs(test_folder / image_path.parent.name)
                
                result_image.save(test_folder / image_path.parent.name / file)
                
                
if __name__ == "__main__":
    
    current_file = Path(__file__)
    path = current_file.parent.parent

    input_folder = path / "test_image"
    test_folder = path / "test"
    
    process_images(input_folder, test_folder)