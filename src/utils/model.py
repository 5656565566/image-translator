from dataclasses import dataclass
from typing import Literal


@dataclass
class TextBlock:
    x1: int
    y1: int
    x2: int
    y2: int
    
    language: Literal["zh", "zh-TW", "jp", "ko", "en"]
    translation: str = ""
    text: str = ""
    translation: str = ""