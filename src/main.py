from ocr import OCR
from inpainting import Inpainting
from rendering import TextEmbedder
from translators import Translator

from PIL import Image
from pathlib import Path

import os

ocr = OCR()
inpainting = Inpainting()
textEmbedder = TextEmbedder()
translator = Translator("http://192.168.100.113:8080/v1/chat/completions")

def process_images(input_folder, test_folder):
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    
    ocr_manga_test = test_folder / "ocr"
    inpainting_test = test_folder / "inpainting"
    rendering_test = test_folder / "rendering"
    output_test = test_folder / "output"
    
    if not os.path.exists(ocr_manga_test):
        os.makedirs(ocr_manga_test)
        
    if not os.path.exists(inpainting_test):
        os.makedirs(inpainting_test)
        
    if not os.path.exists(rendering_test):
        os.makedirs(rendering_test)
        
    if not os.path.exists(output_test):
        os.makedirs(output_test)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                
                image_path = Path(root) / file
                
                image = Image.open(image_path)
                
                if not os.path.exists(ocr_manga_test / image_path.parent.name):
                    os.makedirs(ocr_manga_test / image_path.parent.name)
                
                originals, bboxs = ocr(image, ocr_manga_test / image_path.parent.name / file)
                
                text = "\n".join(originals)
                texts = translator(text)
                
                image = Image.open(image_path)
                output_image, bounding_boxes = inpainting(image, bboxs)
                
                if not os.path.exists(inpainting_test / image_path.parent.name):
                    os.makedirs(inpainting_test / image_path.parent.name)
                
                output_image.save(inpainting_test / image_path.parent.name / file)
                
                if not os.path.exists(output_test / image_path.parent.name):
                    os.makedirs(output_test / image_path.parent.name)
                
                result_image = textEmbedder(texts, output_image, bounding_boxes)
                result_image.save(output_test / image_path.parent.name / file)
                
                
if __name__ == "__main__":
    
    current_file = Path(__file__)
    path = current_file.parent.parent

    input_folder = path / "man"
    test_folder = path / "test"
    
    process_images(input_folder, test_folder)