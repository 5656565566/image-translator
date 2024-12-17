import numpy as np
import cv2
import torch

from io import BytesIO
from typing import Union
from pathlib import Path

from .basemodel import TextDetBase, TextDetBaseDNN
from .db_utils import SegDetectorRepresenter
from .yolov5_utils import non_max_suppression
from .textblock import group_output
from .textmask import refine_mask, refine_undetected_mask

from cuda_test import cuda_test

current_file = Path(__file__)
parent_directory = current_file.parent

REFINEMASK_INPAINT = 0
REFINEMASK_ANNOTATION = 1

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0), auto=False, scaleFill=False, scaleup=True, stride=128):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if not isinstance(new_shape, tuple):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # dw /= 2  # divide padding into 2 sides
    # dh /= 2
    dh, dw = int(dh), int(dw)

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def preprocess_img(img, input_size=(1024, 1024), device='cpu', bgr2rgb=True, half=False, to_tensor=True):
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox(img, new_shape=input_size, auto=False, stride=64)
    if to_tensor:
        img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255
        if to_tensor:
            img_in = torch.from_numpy(img_in).to(device)
            if half:
                img_in = img_in.half()
    return img_in, ratio, int(dw), int(dh)

def postprocess_yolo(det, conf_thresh, nms_thresh, resize_ratio, sort_func=None):
    det = non_max_suppression(det, conf_thresh, nms_thresh)[0]
    if det.device != "cpu":
        det = det.detach_().cpu().numpy()
    det[..., [0, 2]] = det[..., [0, 2]] * resize_ratio[0]
    det[..., [1, 3]] = det[..., [1, 3]] * resize_ratio[1]
    if sort_func is not None:
        det = sort_func(det)

    blines = det[..., 0:4].astype(np.int32)
    confs = np.round(det[..., 4], 3)
    cls = det[..., 5].astype(np.int32)
    return blines, cls, confs

def postprocess_mask(img: Union[torch.Tensor, np.ndarray], thresh=None):
    if isinstance(img, torch.Tensor):
        img = img.squeeze_()
        if img.device != "cpu":
            img = img.detach_().cpu()
        img = img.numpy()
    else:
        img = img.squeeze()
    if thresh is not None:
        img = img > thresh
    img = img * 255

    return img.astype(np.uint8)

class TextDetector:
    lang_list = ["eng", "ja", "unknown"]
    langcls2idx = {"eng": 0, "ja": 1, "unknown": 2}

    def __init__(
        self,
        input_size=1024,
        model_path = parent_directory / "model" / "comictextdetector.pt",
        half=False,
        nms_thresh=0.35,
        conf_thresh=0.4,
        act="leaky"
    ):
        super(TextDetector, self).__init__()

        device = "cuda" if cuda_test() else "cpu"
    
        if device == "cpu":
            model_path = parent_directory / "model" / "comictextdetector.pt.onnx"
        
        if Path(model_path).suffix == ".onnx":
            self.model = cv2.dnn.readNetFromONNX(model_path)
            self.net = TextDetBaseDNN(input_size, model_path)
            self.backend = "opencv"
        else:
            self.net = TextDetBase(model_path, device=device, act=act)
            self.backend = "torch"
        
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.device = device
        self.half = half
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.seg_rep = SegDetectorRepresenter(thresh=0.3)

    @torch.no_grad()
    def predict(self, img, refine_mode=REFINEMASK_INPAINT, keep_undetected_mask=False):

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        
        img_array = np.frombuffer(np.array(img_bytes), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        img_in, ratio, dw, dh = preprocess_img(img, input_size=self.input_size, device=self.device, half=self.half, to_tensor=self.backend=='torch')
        im_h, im_w = img.shape[:2]

        blks, mask, lines_map = self.net(img_in)

        resize_ratio = (im_w / (self.input_size[0] - dw), im_h / (self.input_size[1] - dh))
        blks = postprocess_yolo(blks, self.conf_thresh, self.nms_thresh, resize_ratio)

        if self.backend == "opencv":
            if mask.shape[1] == 2:     # some version of opencv spit out reversed result
                tmp = mask
                mask = lines_map
                lines_map = tmp
        mask = postprocess_mask(mask)

        lines, scores = self.seg_rep(self.input_size, lines_map)
        box_thresh = 0.6
        idx = np.where(scores[0] > box_thresh)
        lines, scores = lines[0][idx], scores[0][idx]
        
        # map output to input img
        mask = mask[: mask.shape[0]-dh, : mask.shape[1]-dw]
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        if lines.size == 0 :
            lines = []
        else :
            lines = lines.astype(np.float64)
            lines[..., 0] *= resize_ratio[0]
            lines[..., 1] *= resize_ratio[1]
            lines = lines.astype(np.int32)
        blk_list = group_output(blks, lines, im_w, im_h, mask)
        mask_refined = refine_mask(img, mask, blk_list, refine_mode=refine_mode)
        if keep_undetected_mask:
            mask_refined = refine_undetected_mask(img, mask, mask_refined, blk_list, refine_mode=refine_mode)
    
        return mask, mask_refined, blk_list
    
if __name__ == "__main__":
    
    def imwrite(img_path, img, ext='.png'):
        suffix = Path(img_path).suffix
        if suffix != '':
            img_path = img_path.replace(suffix, ext)
        else:
            img_path += ext
        cv2.imencode(ext, img)[1].tofile(img_path)
    
    imread = lambda imgpath, read_type=cv2.IMREAD_COLOR: cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), read_type)
    
    model = TextDetector(input_size=1024, act="leaky")
    
    image = imread("test_image/1.png")
    
    mask, mask_refined, blk_list = model.predict(image, refine_mode=REFINEMASK_ANNOTATION, keep_undetected_mask=True)
    
    imwrite("1-mask", mask_refined)