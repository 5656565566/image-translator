import onnxruntime
import numpy as np
import cv2

from PIL import Image
from pathlib import Path

current_file = Path(__file__)
parent_directory = current_file.parent

class EfficientSam:
    
    def __init__(
        self,
        encoder_path: str = parent_directory / "model" / "efficient_sam_vitt_encoder.onnx",
        decoder_path: str = parent_directory / "model" / "efficient_sam_vitt_decoder.onnx",
        session_path: str = parent_directory / "model" / "efficient_sam_vitt.onnx"
    ):
        self.session = onnxruntime.InferenceSession(session_path)
        self.encoder_session = onnxruntime.InferenceSession(encoder_path)
        self.decoder_session = onnxruntime.InferenceSession(decoder_path)
    
    def predict(self, image: Image, input_points: list):
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
        
        combined_mask = np.logical_or.reduce(masks)
        output_image = (input_image[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        output_image[combined_mask] = [255, 255, 255]
        
        return Image.fromarray(output_image), bounding_boxes
    
    def test_predict(self, image: Image, input_points: list):
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
        
        combined_mask = np.logical_or.reduce(masks)
        output_image = (input_image[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        output_image[combined_mask] = [255, 255, 255]
        
        # 标红内接框
        for bbox in bounding_boxes:
            x1, y1, x2, y2 = bbox
            print(f"[{x1}, {y1}, {x2}, {y2}],")
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        return output_image
    

if __name__ == "__main__":

    efficient_sam = EfficientSam(
        "model/efficient_sam_vitt_encoder.onnx",
        "model/efficient_sam_vitt_decoder.onnx",
        "model/efficient_sam_vitt.onnx",
        )

    image = Image.open("1.png")

    points = [
        [[86, -11], [288, 74]],
        [[1116, 119], [1155, 128]],
        [[611, 211], [659, 371]],
        [[437, 340], [500, 571]],
        [[46, 591], [119, 718]],
        [[213, 834], [272, 1027]],
        [[917, 1044], [1282, 1071]],
        [[407, 995], [696, 1077]],
        [[746, 496], [1477, 1023]],
    ]

    masks = efficient_sam.test_predict(
        image, points
    )

    import imgviz

    imgviz.io.imsave(f"mask.png", masks)