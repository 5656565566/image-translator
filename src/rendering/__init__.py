from .easy import EasyTextEmbedder as EasyTextEmbedder

from PIL import Image

class TextEmbedder:
    def __init__(self, name: str):
        self.name = name
        
        if name == "easy":
            self.easyTextEmbedder = EasyTextEmbedder(
                "MiSans-Normal.ttf"
            )

    def __call__(
        self,
        texts: list[str],
        image: Image,
        bounding_boxes: list
    ) -> Image:
        if self.name == "easy":
            return self.easyTextEmbedder.embed_text(
                texts, image, bounding_boxes
            )