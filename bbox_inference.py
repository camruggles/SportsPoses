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
import json
import datetime
import numpy as np
from math import sqrt


# IMPLEMENTATION NOTES: NOT MEANT TO PASS STANDARDS
'''
the bounding box visualization and the coordinates of the bounding boxes are not consistent
area < 1000 could work
m by n, y by x

if the second largest bounding box has a higher confidence score

filter out non central boxes
weighted score of size and confidence


size * confidence score squared
raise the bottom bound
require a min height or width 



=======================

look for possible ways to isolate bad samples

either right swipe 
or
left swipe plus click on correction or mark a new box
'''
# IMPLEMENTATION NOTES: NOT MEANT TO PASS STANDARDS


def area(box):
    ''' box is a tensor'''
    [[x1,y1,x2,y2]] = box
    return (x2-x1) * (y2-y1)


def render_and_wait(predictions, box=None):
    '''
    args:
    predictions, the instances object that contains the outcome of the model inference
    box: a tensor and/or an instance object, that contains the bounding box coordinates

    returns:
    the key code from cv2 waitKey key press
    '''
    #visualize the bounding box and other predictions
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(predictions)

    img = out.get_image()[:, :, ::-1]
    img = np.ascontiguousarray(img, dtype=np.uint8)

    # draw rectangle around center of the image
    m,n,c = img.shape
    midpoint_img_y = m/2
    midpoint_img_x = n/2

    left_bound = midpoint_img_x - n/4 +5
    right_bound = midpoint_img_x + n/4 -5
    top_bound = midpoint_img_y + m/4 -5
    bottom_bound = midpoint_img_y - m/ +5

    start_point = ( int(left_bound), int(bottom_bound))
    end_point = (int(right_bound), int(top_bound) )

    color = (255,0,0)
    thickness=2
    img2 = cv2.rectangle(img, start_point, end_point, color, thickness)

    # draw the bounding box coordinates into the image separate from the visualization
    if box is not None:
        
        [[x1,y1,x2,y2]] = box
        
        start_point = ( int(x1), int(y1))
        end_point = (int(x2), int(y2) )

        color = (0,255,0)
        thickness=2
        img2 = cv2.rectangle(img, start_point, end_point, color, thickness)


    # show image and wait for a key
    cv2.imshow('pane', img2)
    key = cv2.waitKey(0)
    # pdb.set_trace()
    print(key)
    return key

def near_center(box, m, n, score):
    '''
    args
    box: tensor, contains the bounding box coordinates
    m: int, is the height of the image
    n: int, is the width of the image
    score: float, is the confidence score of model inference

    returns: bool, float
    bool is whether or not the bbox center is near the center of the image
    float is the distance from the bbox center to the image center, normalized by distance from image center to image corner
    '''

    # compute bounds
    midpoint_img_y = m/2
    midpoint_img_x = n/2
    left_bound = midpoint_img_x - n/4 +5
    right_bound = midpoint_img_x + n/4 -5
    top_bound = midpoint_img_y + m/4 -5
    bottom_bound = midpoint_img_y - m/4 +5

    #compute box center
    [[x1,y1,x2,y2]] = box
    box_center_x = (x1+x2)/2
    box_center_y = (y1+y2)/2

    # debug information
    print("bounds")
    print(box)
    print(score)
    print(left_bound, right_bound)
    print(box_center_x)
    print(bottom_bound, top_bound)
    print(box_center_y)
    print()

    # compute bbox center distance to image center, normalized
    pythagorean_distance = sqrt((box_center_x - midpoint_img_x)**2 + (box_center_y- midpoint_img_y)**2)
    total_img_pythagorean = sqrt((n - midpoint_img_x)**2 + (m - midpoint_img_y)**2)
    center_dist_normalized = pythagorean_distance / total_img_pythagorean

    # see if bbox center is within bounds
    flag1 = False
    flag2 = False

    if left_bound < box_center_x and box_center_x < right_bound:
        flag1=True
    if bottom_bound < box_center_y and box_center_y < top_bound:
        flag2=True
    print(flag1, flag2)
    if flag1 and flag2:
        return True, center_dist_normalized
    return False, center_dist_normalized
    

def write(good_boxes):
    '''
    dumps contents of good_boxes to a json, named after the time of program execution to the second

    args:
    good_boxes, a list of dictionaries containing the image name and the desired bounding box to be used for later

    returns:
    none
    '''
    sec=datetime.datetime.now().second
    minute = datetime.datetime.now().minute
    hour= datetime.datetime.now().hour
    day= datetime.datetime.now().day
    month= datetime.datetime.now().month
    filename = "boxes_{}_{}_{}_{}_{}.json".format(month,day,hour,minute,sec)
    with open(filename, "w") as fp:
        json.dump(good_boxes, fp)


# load in images
imgs = []
names = []
print(len(os.listdir("./imgs")))
for f in os.listdir("./imgs")[0:200]:
    filepath = os.path.join("./imgs", f)
    im = cv2.imread(filepath)
    imgs.append(im)
    names.append(filepath)

# load the model

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
#predictor = build_model(cfg)



good_boxes = []



for i in range(len(imgs)):
    # perform inference
    im = imgs[i]
    print(im.shape)
    m,n,c = im.shape
    outputs = predictor(im)

    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    print(names[i])

    flag =False
    
    # get the bounding box with the best criteria for containing the gymnast
    maxArea = -1
    maxIndex= -1

    maxScore = 0.0
    if len(outputs['instances'])>0:
        # find the largest box
        # pdb.set_trace()
        print("=============================================================")
        for j in range(len(outputs['instances'])):
            # if not human, disqualify
            if outputs['instances'][j].pred_classes != 0:
                continue
            box = outputs['instances'][j].pred_boxes
            score = outputs['instances'][j].scores
            
            # see if it's near center, disqualify if on edges of image
            centered, center_distance = near_center(box, m, n, score)
            if not centered:
                # pass
                continue
            
            # if area(box) > maxArea:# and score >= 0.7:
            #     maxArea = area(box)
            #     maxIndex=j

            # compute criteria and determine if it's the best
            print("========")
            tempScore = area(box) * score * score
            print(j)
            print(area(box))
            print(score)
            print(center_distance, 1-center_distance)
            print(tempScore)

            if tempScore > maxScore:
                maxScore = tempScore
                maxIndex=j
                print("max index set")
            print()
        
        if maxIndex != -1:
            flag=True

    
    if flag:
        # We can use `Visualizer` to draw the predictions on the image.

        # visualize the best bbox
        predictions = outputs['instances'][maxIndex].to("cpu")
        
        print(area(predictions.pred_boxes))
        box = predictions.pred_boxes
        key = render_and_wait(predictions, box)

        # if alt key, quit
        if key == 233:
            write(good_boxes)
            quit()
        # if 0 num key, save as a good inference
        if key == 48:
            [[x1,y1,x2,y2]] = predictions.pred_boxes
            good_boxes.append({"filename": names[i], "x1": x1.item(), "y1": y1.item(), "x2":x2.item(), "y2":y2.item()  })
        # if w key, reshow the image with all the bounding boxes from inference to debug
        if key == 119:
            predictions = outputs['instances'].to("cpu")
            key = render_and_wait(predictions)
            # if alt key, quit
            if key == 233:
                write(good_boxes)
                quit()

        print()
    flag=False

# save results
write(good_boxes)
