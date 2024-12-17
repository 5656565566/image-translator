from .sakura import SakuraTranslator
from deep_translator import (
    GoogleTranslator,
    ChatGptTranslator,
    MicrosoftTranslator,
    PonsTranslator,
    LingueeTranslator,
    MyMemoryTranslator,
    YandexTranslator,
    PapagoTranslator,
    DeeplTranslator,
    QcriTranslator,
    single_detection,
    batch_detection
)

models = {
    "SakuraTranslator" : SakuraTranslator,
    "GoogleTranslator" : GoogleTranslator,
    "ChatGptTranslator": ChatGptTranslator,
    "MicrosoftTranslator" : MicrosoftTranslator,
    "PonsTranslator" : PonsTranslator,
    "LingueeTranslator" : LingueeTranslator,
    "MyMemoryTranslator" : MyMemoryTranslator,
    "YandexTranslator" : YandexTranslator,
    "PapagoTranslator" : PapagoTranslator,
    "DeeplTranslator" : DeeplTranslator,
    "QcriTranslator" : QcriTranslator,
}

class Translator:
    def __init__(self, name: str, **kwargs):
        self.name = name
        
        self.model = models.get(name)(**kwargs)

    def __call__(
        self,
        source_text: list,
    ) -> list[str]:
        if self.name == "SakuraTranslator":
            translated_text = self.model.translate("\n".join(source_text))
            return translated_text.split("\n")

        elif self.name == "single_detection":
            
            translated_text = []
            
            for source in source_text:
                translated_text.append(single_detection(source_text))
            
            return translated_text
        
        elif self.name == "batch_detection":
            return batch_detection(source_text)
        
        else:
            
            translated_text = []
            
            for source in source_text:
                translated_text.append(self.model.translate(text=source))
                
            return translated_text