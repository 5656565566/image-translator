from easyocr import Reader


class EasyOCR:
    def __init__(
        self,
        languages=["en"],
        gpu=True
    ):
        self.readers = []

        for language in languages:
            self.readers.append(Reader([language], gpu=gpu))

    def __call__(self, image, detail=1):
        results = {}

        for reader in self.readers:
            language = reader.lang_list[0]
            results[language] = reader.readtext(image, detail=detail)

        return results


if __name__ == "__main__":
    ocr = EasyOCR(languages=["ja", "en"])

    image_path = "test_image/1.png"
    result = ocr(image_path)

    print(result)
