from pathlib import Path
from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel, logging as transformers_logging

import easyocr
import jaconv
import re

transformers_logging.set_verbosity_error()

current_file = Path(__file__)
parent_directory = current_file.parent

class MangaOCR:
    def __init__(
        self,
        # model_path="kha-white/manga-ocr-base",
        model_path=parent_directory / "model",
        languages=["ja"],
        proxies={'http': 'http://127.0.0.1:7890',
                 'https': 'http://127.0.0.1:7890'},
    ):

        self.readers = []

        for language in languages:
            self.readers.append(easyocr.Reader([language]))

        self.pretrained_model_path = model_path
        self.processor = ViTImageProcessor.from_pretrained(
            self.pretrained_model_path, proxies=proxies)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_path, proxies=proxies)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.pretrained_model_path, proxies=proxies)
        self.model.cuda()

    def preprocess(self, img):
        pixel_values = self.processor(img, return_tensors="pt").pixel_values
        return pixel_values.squeeze()

    def post_process(self, text):
        
        text = "".join(text.split())
        text = text.replace("…", "...")
        text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
        text = jaconv.h2z(text, ascii=True, digit=True)
        return text

    def ocr_manga(self, image):
        
        x = self.preprocess(image)
        x = self.model.generate(x[None].to(
            self.model.device), max_length=300)[0].cpu()
        x = self.tokenizer.decode(x, skip_special_tokens=True)
        x = self.post_process(x)
        return x
