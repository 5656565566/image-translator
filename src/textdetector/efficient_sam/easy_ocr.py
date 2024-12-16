from pathlib import Path

import easyocr
import jaconv
import re
import numpy as np

current_file = Path(__file__)
parent_directory = current_file.parent

class EasyOcr:
    def __init__(
        self,
        languages=["ja"],
        threshold=30,
        distance_threshold=15,
        text_threshold=0.8
    ):

        self.readers = []

        for language in languages:
            self.readers.append(easyocr.Reader([language]))

        self.threshold = threshold
        self.distance_threshold = distance_threshold
        self.text_threshold = text_threshold

    def post_process(self, text):
        text = "".join(text.split())
        text = text.replace("…", "...")
        text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
        text = jaconv.h2z(text, ascii=True, digit=True)
        return text

    def detect_text_boxes(self, image):
        image_np = np.array(image)

        dialogue_boxes = []

        for reader in self.readers:
            result = reader.detect(
                image_np, text_threshold=self.text_threshold)[0][0]
            for bbox in result:
                x_min, y_min, x_max, y_max = bbox[0], bbox[2], bbox[1], bbox[3]
                dialogue_boxes.append((x_min, y_min, x_max, y_max))

        return dialogue_boxes

    def merge_overlapping_boxes(self, boxes):
        merged = True
        while merged:
            merged = False
            new_boxes = []
            while boxes:
                current_box = boxes.pop(0)
                x_min, y_min, x_max, y_max = current_box

                has_overlap = False
                for i, box in enumerate(boxes):
                    bx_min, by_min, bx_max, by_max = box
                    if not (x_max < bx_min or x_min > bx_max or y_max < by_min or y_min > by_max):
                        x_min = min(x_min, bx_min)
                        y_min = min(y_min, by_min)
                        x_max = max(x_max, bx_max)
                        y_max = max(y_max, by_max)
                        boxes.pop(i)
                        has_overlap = True
                        merged = True
                        break

                if not has_overlap:
                    new_boxes.append((x_min, y_min, x_max, y_max))
                else:
                    boxes.append((x_min, y_min, x_max, y_max))

            boxes = new_boxes

        return boxes

    def merge_nearby_boxes(self, boxes):
        def are_boxes_near(box1, box2, threshold):
            x_min1, y_min1, x_max1, y_max1 = box1
            x_min2, y_min2, x_max2, y_max2 = box2

            horizontal_dist = max(0, max(x_min2 - x_max1, x_min1 - x_max2))
            vertical_dist = max(0, max(y_min2 - y_max1, y_min1 - y_max2))

            return horizontal_dist <= threshold and vertical_dist <= threshold

        merged = True
        while merged:
            merged = False
            new_boxes = []
            while boxes:
                current_box = boxes.pop(0)
                x_min, y_min, x_max, y_max = current_box

                has_nearby = False
                for i, box in enumerate(boxes):
                    if are_boxes_near(current_box, box, self.distance_threshold):
                        x_min = min(x_min, box[0])
                        y_min = min(y_min, box[1])
                        x_max = max(x_max, box[2])
                        y_max = max(y_max, box[3])
                        boxes.pop(i)
                        has_nearby = True
                        merged = True
                        break

                if not has_nearby:
                    new_boxes.append((x_min, y_min, x_max, y_max))
                else:
                    boxes.append((x_min, y_min, x_max, y_max))

            boxes = new_boxes

        return boxes

    def merge_boxes(self, boxes, expansion=True):
        merged_boxes = []
        while boxes:
            current_box = boxes.pop(0)
            x_min, y_min, x_max, y_max = current_box

            width = x_max - x_min
            height = y_max - y_min

            merge_candidate = []
            for box in boxes[:]:
                bx_min, by_min, bx_max, by_max = box
                if (abs(bx_min - x_max) < self.threshold or abs(bx_max - x_min) < self.threshold) and \
                        (abs(by_min - y_max) < self.threshold or abs(by_max - y_min) < self.threshold):
                    merge_candidate.append(box)
                    boxes.remove(box)

            if merge_candidate:
                for box in merge_candidate:
                    bx_min, by_min, bx_max, by_max = box
                    x_min = min(x_min, bx_min)
                    y_min = min(y_min, by_min)
                    x_max = max(x_max, bx_max)
                    y_max = max(y_max, by_max)

            if expansion:
                x_min = int(x_min - width * 0.03)
                y_min = int(y_min - height * 0.07)
                x_max = int(x_max + width * 0.03)
                y_max = int(y_max + height * 0.07)

            merged_boxes.append((x_min, y_min, x_max, y_max))

        return merged_boxes

    def getBboxs(self, image):
        dialogue_boxes = self.detect_text_boxes(image)

        merged_boxes = self.merge_boxes(dialogue_boxes)
        merged_boxes = self.merge_nearby_boxes(merged_boxes)
        merged_boxes = self.merge_overlapping_boxes(merged_boxes)

        bboxs = []
        for box in merged_boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            bboxs.append([(x_min, y_min), (x_max, y_max)])
            
            
        return bboxs