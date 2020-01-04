#例如我们选择keras yolo3进行文字检测，选择pytorch进行文字识别，去掉文字方向检测（假定输入的图片绝大多数是方向正确的），
# 那么即可对chineseocr的源代码进行大幅精简。在model.py代码的基础上进行修改，去繁存简，
# 对识别能力进行封装，方便提供给其它应用程序使用。修改后的核心代码如下：
# 文字检测
from config import IMGSIZE
from crnn.crnn_torch import crnnOcr as crnnOcr
import cv2
import numpy as np
from PIL import Image
from text.detector.detectors import TextDetector
from apphelper.image import get_boxes,letterbox_image
from apphelper.image import estimate_skew_angle ,rotate_cut_img,xy_rotate_box,sort_box,box_rotate,solve
from text import keras_detect as detect

# 文字检测
def text_detect(img,MAX_HORIZONTAL_GAP=30,MIN_V_OVERLAPS=0.6,MIN_SIZE_SIM=0.6,TEXT_PROPOSALS_MIN_SCORE=0.7,TEXT_PROPOSALS_NMS_THRESH=0.3,TEXT_LINE_NMS_THRESH=0.3,):
    boxes, scores = detect.text_detect(np.array(img))
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    textdetector = TextDetector(MAX_HORIZONTAL_GAP, MIN_V_OVERLAPS, MIN_SIZE_SIM)
    shape = img.shape[:2]
    boxes = textdetector.detect(boxes,scores[:, np.newaxis],shape,TEXT_PROPOSALS_MIN_SCORE,TEXT_PROPOSALS_NMS_THRESH,TEXT_LINE_NMS_THRESH,)
    text_recs = get_boxes(boxes)
    newBox = []
    rx = 1
    ry = 1
    for box in text_recs:
        x1, y1 = (box[0], box[1])
        x2, y2 = (box[2], box[3])
        x3, y3 = (box[6], box[7])
        x4, y4 = (box[4], box[5])
        newBox.append([x1 * rx, y1 * ry, x2 * rx, y2 * ry, x3 * rx, y3 * ry, x4 * rx, y4 * ry])
    return newBox

# 文字识别
def crnnRec(im, boxes, leftAdjust=False, rightAdjust=False, alph=0.2, f=1.0):
    results = []
    im = Image.fromarray(im)
    for index, box in enumerate(boxes):
        degree, w, h, cx, cy = solve(box)
        partImg, newW, newH = rotate_cut_img(im, degree, box, w, h, leftAdjust, rightAdjust, alph)
        text = crnnOcr(partImg.convert('L'))
        if text.strip() != u'':
            results.append({'cx': cx * f, 'cy': cy * f, 'text': text, 'w': newW * f, 'h': newH * f,
                            'degree': degree * 180.0 / np.pi})
    return results

# 文字检测、文字识别的能力封装
def ocr_model(img, leftAdjust=True, rightAdjust=True, alph=0.02):
    img, f = letterbox_image(Image.fromarray(img), IMGSIZE)
    img = np.array(img)
    config = dict(MAX_HORIZONTAL_GAP=50,  ##字符之间的最大间隔，用于文本行的合并
                  MIN_V_OVERLAPS=0.6,
                  MIN_SIZE_SIM=0.6,
                  TEXT_PROPOSALS_MIN_SCORE=0.1,
                  TEXT_PROPOSALS_NMS_THRESH=0.3,
                  TEXT_LINE_NMS_THRESH=0.7,  ##文本行之间测iou值
                  )
    config['img'] = img
    text_recs = text_detect(**config)  ##文字检测
    newBox = sort_box(text_recs)  ##行文本识别
    result = crnnRec(np.array(img), newBox, leftAdjust, rightAdjust, alph, 1.0 / f)
    return result