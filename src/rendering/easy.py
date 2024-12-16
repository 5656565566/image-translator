from PIL import Image, ImageDraw, ImageFont


class EasyTextEmbedder:
    def __init__(
        self,
        font_path: str = "MiSans-Normal.ttf",
        max_font_size: int = 100
    ):
        """
        初始化 TextEmbedder 类
        """
        self.font_path = font_path
        self.max_font_size = max_font_size

    def fit_font_size(self, text: str, box_width: int, box_height: int, vertical: bool) -> int:
        """
        根据文本框的大小动态计算合适的字体大小
        """
        min_font_size = 1
        max_font_size = self.max_font_size

        while min_font_size <= max_font_size:
            mid_font_size = (min_font_size + max_font_size) // 2
            font = ImageFont.truetype(self.font_path, size=mid_font_size)

            char_bbox = font.getbbox("测")  # 假定一个字，支持竖排或横排
            char_width = max(1, char_bbox[2] - char_bbox[0])  # 避免 char_width 为 0
            char_height = max(1, char_bbox[3] - char_bbox[1])  # 避免 char_height 为 0

            if vertical:
                # 竖排时，每列的宽度和文本总高度
                max_chars_per_column = max(1, box_height // char_height)
                text_width = char_width * ((len(text) + max_chars_per_column - 1) // max_chars_per_column)
                text_height = char_height * min(len(text), max_chars_per_column)
            else:
                # 横排时，每行的高度和文本总宽度
                max_chars_per_row = max(1, box_width // char_width)
                text_height = char_height * ((len(text) + max_chars_per_row - 1) // max_chars_per_row)
                text_width = char_width * min(len(text), max_chars_per_row)

            if text_width <= box_width and text_height <= box_height:
                min_font_size = mid_font_size + 1
            else:
                max_font_size = mid_font_size - 1

        return max_font_size

    def embed_text(
        self, texts: list, image: Image, bounding_boxes: list
    ) -> Image:
        """
        在图像的 bounding_boxes 中嵌入文本
        """
        
        draw = ImageDraw.Draw(image)

        for box, text in zip(bounding_boxes, texts):
            x1, y1, x2, y2 = box
            # 内框缩小
            box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
            margin_x, margin_y = int(box_width * 0.2), int(box_height * 0.2)
            x1, y1 = x1 + margin_x, y1 + margin_y
            x2, y2 = x2 - margin_x, y2 - margin_y

            # 判断横排或竖排
            vertical = box_height > box_width

            # 动态计算字体大小
            font_size = self.fit_font_size(text, x2 - x1, y2 - y1, vertical)
            font = ImageFont.truetype(self.font_path, size=font_size)

            if vertical:
                # 竖排逻辑：按列排列
                char_bbox = font.getbbox("测")
                char_height = char_bbox[3] - char_bbox[1]
                char_width = char_bbox[2] - char_bbox[0]
                max_chars_per_column = (y2 - y1) // char_height
                current_x, current_y = x1, y1

                for i, char in enumerate(text):
                    if current_y + char_height > y2:
                        current_x += char_width  # 换列
                        current_y = y1
                    draw.text((current_x, current_y), char, fill="black", font=font)
                    current_y += char_height
            else:
                # 横排逻辑：按行排列
                char_bbox = font.getbbox("测")
                char_width = char_bbox[2] - char_bbox[0]
                char_height = char_bbox[3] - char_bbox[1]
                max_chars_per_row = (x2 - x1) // char_width
                current_x, current_y = x1, y1

                for i, char in enumerate(text):
                    if current_x + char_width > x2:
                        current_y += char_height  # 换行
                        current_x = x1
                    draw.text((current_x, current_y), char, fill="black", font=font)
                    current_x += char_width

        return image


if __name__ == "__main__":
    # 示例使用
    # 假设 predict 返回的图像和框
    predict_image = Image.open("mask.png")
    predict_bounding_boxes = [
        [0, 0, 379, 81],
        [1114, 113, 1154, 130],
        [611, 217, 695, 555],
        [401, 297, 529, 618],
        [12, 560, 138, 735],
        [178, 822, 312, 1033],
        [929, 1048, 1298, 1079],
        [411, 963, 695, 1079],
        [741, 517, 1479, 1079]
    ]  # 示例框

    # 创建 TextEmbedder
    text_embedder = EasyTextEmbedder(font_path="MiSans-Normal.ttf")

    # 嵌入文本
    texts = [
    "",
    "",
    "拯救了土河！",
    "加奈！！",
    "0日元微笑？",
    "在上空预先配置。。。！？",
    "虽然是服务的客人使用的，但是客人很在意",
    "原作：等待岩田话的原作・岩田雪花・作画・青木裕",
    "神谕岩出雪内麻林粉/失败注麻少工梨快爱饲绘",
    ]
    result_image = text_embedder.embed_text(
        texts=texts, image=predict_image, bounding_boxes=predict_bounding_boxes
    )

    # 保存结果
    result_image.save("output_image.jpg")