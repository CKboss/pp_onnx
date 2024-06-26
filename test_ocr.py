import cv2
import time
from pp_onnx.onnx_paddleocr import ONNXPaddleOcr,draw_ocr
import sys


model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)

def sav2Img(org_img, result, name="./result_img/draw_ocr_996_1.jpg"):
    # 显示结果
    from PIL import Image
    result = result[0]
    # image = Image.open(img_path).convert('RGB')
    # 图像转BGR2RGB
    image = org_img[:, :, ::-1]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores)
    im_show = Image.fromarray(im_show)
    im_show.save(name)

img = cv2.imread('./test_img/1.jpg')
s = time.time()
result = model.ocr(img)
e = time.time()
print("total time: {:.3f}".format(e - s))
print("result:", result)
for box in result[0]:
    print(box)
sav2Img(img, result)