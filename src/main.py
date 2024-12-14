from ocr import MangaOCR
from inpainting import EfficientSam
from rendering import TextEmbedder
from translators import LlamaCppTranslator

from PIL import Image
from pathlib import Path

import os

mangaOCR = MangaOCR()
efficientSam = EfficientSam()
textEmbedder = TextEmbedder()
translator = LlamaCppTranslator("http://192.168.100.113:8080/v1/chat/completions")

def process_images(input_folder, test_folder):
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    
    ocr_manga_test = test_folder / "ocr_manga"
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

    for _, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                image = Image.open(input_folder / file)
                originals, bboxs = mangaOCR.ocr_manga(image, ocr_manga_test / file)
                
                text = "\n".join(originals)
                texts = translator.translate(text).replace("翻译结果：", "").split("\n")
                
                image = Image.open(input_folder / file)
                output_image, bounding_boxes = efficientSam.predict(image, bboxs)
                output_image.save(inpainting_test / file)
                
                result_image = textEmbedder.embed_text(texts, output_image, bounding_boxes)
                result_image.save(output_test / file)
                
                
if __name__ == "__main__":
    
    current_file = Path(__file__)
    path = current_file.parent.parent

    input_folder = path / "test_image"
    test_folder = path / "test"
    
    process_images(input_folder, test_folder)