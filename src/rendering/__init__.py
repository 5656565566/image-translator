from .easy import EasyTextEmbedder as EasyTextEmbedder

from PIL import Image

class TextEmbedder:
    def __init__(self):
        pass

    def __call__(
        self,
        texts: list[str],
        image: Image,
        bounding_boxes: list
    ):
        pass