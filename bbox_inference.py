from detectron2.utils.events import EventStorage
from detectron2 import model_zoo
from detectron2.config import get_cfg
import cv2
import pdb
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
import os
import torch

# m by n, y by x
def area(box):
    [[x1,y1,x2,y2]] = box
    return (x2-x1) * (y2-y1)

    
def render_and_wait(predictions):
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(predictions)

    cv2.imshow('pane', out.get_image()[:, :, ::-1])
    key = cv2.waitKey(0)
    print(key)
    if key == 233:
        quit()
    return key

def near_center(box, m, n):
    midpoint_img_y = m/2
    midpoint_img_x = n/2
    left_bound = midpoint_img_x - n/4 +5
    right_bound = midpoint_img_x + n/4 -5
    top_bound = midpoint_img_y + m/4 -5
    bottom_bound = midpoint_img_y - m/4 +5
    [[x1,y1,x2,y2]] = box
    box_center_x = (x1+x2)/2
    box_center_y = (y1+y2)/2
    flag1 = False
    flag2 = False
    print("bounds")
    print(left_bound, right_bound)
    print(box_center_x)
    print(bottom_bound, top_bound)
    print(box_center_y)
    if left_bound < box_center_x and box_center_x < right_bound:
        flag1=True
    if bottom_bound < box_center_y and box_center_y < top_bound:
        flag2=True
    print(flag1, flag2)
    if flag1 and flag2:
        return True
    


imgs = []
names = []
for f in os.listdir("./imgs")[:100]:
    filepath = os.path.join("./imgs", f)
    im = cv2.imread(filepath)
    imgs.append(im)
    names.append(filepath)


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
#predictor = build_model(cfg)

for i in range(len(imgs)):
    im = imgs[i]
    print(im.shape)
    m,n,c = im.shape
    outputs = predictor(im)

    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    print(names[i])

    flag =False
    
    maxArea = -1
    maxIndex= -1
    if len(outputs['instances'])>0:
        # find the largest box
        # pdb.set_trace()
        for j in range(len(outputs['instances'])):
            if outputs['instances'][j].pred_classes != 0:
                continue
            box = outputs['instances'][j].pred_boxes
            score = outputs['instances'][j].scores
            if area(box) > maxArea and score >= 0.7:
                maxArea = area(box)
                maxIndex=j
        
        if maxIndex != -1:
            flag=True

    
    if flag:
        # We can use `Visualizer` to draw the predictions on the image.
        predictions = outputs['instances'][maxIndex].to("cpu")
        
        print(area(predictions.pred_boxes))
        if not near_center(predictions.pred_boxes, m, n):
            continue
        key = render_and_wait(predictions)
        if key == 119:
            predictions = outputs['instances'].to("cpu")
            _ = render_and_wait(predictions)

        print()
    flag=False

    '''
    with EventStorage() as storage:
        model.eval()
    '''


# area < 1000 could work