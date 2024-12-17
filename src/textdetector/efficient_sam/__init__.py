import onnxruntime
import numpy as np

from PIL import Image, ImageDraw
from pathlib import Path

from .easy_ocr import EasyOcr

current_file = Path(__file__)
parent_directory = current_file.parent

class EfficientSam:
    
    def __init__(
        self,
        encoder_path: str = parent_directory / "model" / "efficient_sam_vitt_encoder.onnx",
        decoder_path: str = parent_directory / "model" / "efficient_sam_vitt_decoder.onnx",
        session_path: str = parent_directory / "model" / "efficient_sam_vitt.onnx",
        languages=["ja"],
    ):
        self.session = onnxruntime.InferenceSession(session_path)
        self.encoder_session = onnxruntime.InferenceSession(encoder_path)
        self.decoder_session = onnxruntime.InferenceSession(decoder_path)

        self.easyocr = EasyOcr(languages)
    
    def predict(self, image: Image):
        
        input_points = self.easyocr.getBboxs(image)
        
        input_image = np.array(image)
        input_image = input_image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        
        image_embeddings, = self.encoder_session.run(
            output_names=None,
            input_feed={"batched_images": input_image},
        )
        
        masks = []
        input_labels = np.array([[[2, 1]]], dtype=np.float32)
        
        bounding_boxes = []  # 用于存储内接框的坐标
        
        for point in input_points:
            input_point = np.array([[point]], dtype=np.float32)
        
            predicted_logits, _, _ = self.decoder_session.run(
                output_names=None,
                input_feed={
                    "image_embeddings": image_embeddings,
                    "batched_point_coords": input_point,
                    "batched_point_labels": input_labels,
                    "orig_im_size": np.array(input_image.shape[2:], dtype=np.int64),
                },
            )
        
            mask = predicted_logits[0, 0, 0, :, :] >= 0
            masks.append(mask)
            
            # 计算内接框 (Bounding Box)
            mask_region = np.where(mask)
            y1, x1 = np.min(mask_region[0]), np.min(mask_region[1])
            y2, x2 = np.max(mask_region[0]), np.max(mask_region[1])
            bounding_boxes.append([x1, y1, x2, y2])    
        
        return bounding_boxes, masks

if __name__ == "__main__":

    efficient_sam = EfficientSam()

    image = Image.open("test_image/1.png")

    bounding_boxes, masks = efficient_sam.predict(image)

    for bounding_boxe in bounding_boxes:
        ImageDraw.Draw(image)
    
    input_image = np.array(image)
    output_image = (input_image[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    combined_mask = np.logical_or.reduce(masks)
    output_image[combined_mask] = [255, 255, 255]
    
    Image.fromarray(output_image).save("test/mask.png")