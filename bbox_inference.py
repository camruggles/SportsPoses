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


'''
if space, good
if w, show more boxes
r, click new box
d, draw new box

space 32
w 119
r 114
d 100
alt 233
backspace 8

'''
# IMPLEMENTATION NOTES: NOT MEANT TO PASS STANDARDS




W_KEY = 119
R_KEY = 114
D_KEY = 100
BACKSPACE_KEY = 8
ALT_KEY = 233
SPACE_KEY = 32

SAVEPATH = './boxes'



mouseX = [-1]
mouseY = [-1]



def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print('click', x,y)
        # cv2.circle(img2,(x,y),100,(255,0,0),-1)
        mouseX.append(x)
        mouseY.append(y)

def box_area(box):
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
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(config.DATASETS.TRAIN[0]), scale=1.0) # big bug here, make scale 1.0
    out = v.draw_instance_predictions(predictions)

    img = out.get_image()[:, :, ::-1]
    img = np.ascontiguousarray(img, dtype=np.uint8)

    # draw rectangle around center of the image
    height,width,c = img.shape
    midpoint_img_y = height/2
    midpoint_img_x = width/2

    left_bound = midpoint_img_x - width/4 +5
    right_bound = midpoint_img_x + width/4 -5
    top_bound = midpoint_img_y + height/4 -5
    bottom_bound = midpoint_img_y - height/ +5

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
    #print(key)
    return key

def near_center(box, height, width, score):
    '''
    args
    box: tensor, contains the bounding box coordinates
    height: int, is the height of the image
    width: int, is the width of the image
    score: float, is the confidence score of model inference

    returns: bool, float
    bool is whether or not the bbox center is near the center of the image
    float is the distance from the bbox center to the image center, normalized by distance from image center to image corner
    '''

    # compute bounds
    midpoint_img_y = height/2
    midpoint_img_x = width/2
    left_bound = midpoint_img_x - width/4 +5
    right_bound = midpoint_img_x + width/4 -5
    top_bound = midpoint_img_y + height/4 -5
    bottom_bound = midpoint_img_y - height/4 +5

    #compute box center
    [[x1,y1,x2,y2]] = box
    box_center_x = (x1+x2)/2
    box_center_y = (y1+y2)/2

    # debug information
    # #print("bounds")
    # #print(box)
    # #print(score)
    # #print(left_bound, right_bound)
    # #print(box_center_x)
    # #print(bottom_bound, top_bound)
    # #print(box_center_y)
    # #print()

    # compute bbox center distance to image center, normalized
    pythagorean_distance = sqrt((box_center_x - midpoint_img_x)**2 + (box_center_y- midpoint_img_y)**2)
    total_img_pythagorean = sqrt((width - midpoint_img_x)**2 + (height - midpoint_img_y)**2)
    center_dist_normalized = pythagorean_distance / total_img_pythagorean

    # see if bbox center is within bounds
    x_contained_flag = False
    y_contained_flag = False

    if left_bound < box_center_x and box_center_x < right_bound:
        x_contained_flag=True
    if bottom_bound < box_center_y and box_center_y < top_bound:
        y_contained_flag=True
    # debug information
    # #print(x_contained_flag, y_contained_flag)
    if x_contained_flag and y_contained_flag:
        return True, center_dist_normalized
    return False, center_dist_normalized


def write_to_json(good_boxes, imgid = None):
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
    if imgid is None:
        filename = "boxes_{}_{}_{}_{}_{}.json".format(month,day,hour,minute,sec)
    else:
        filename = "boxes_{}.json".format(imgid)
    #print("writing to ", filename)
    with open(os.path.join(SAVEPATH, filename), "w") as fp:
        json.dump(good_boxes, fp)

BOX_PADDING = 10
COUNTER= 16

# skipped 14-18
# complete 0-13, 19, 
# MAX 88
# load in images
imgs = []
names = []
#print(len(os.listdir("./imgs")))
i1 = COUNTER*206
i2 = (COUNTER+1) * 206 + 1
#print(i1,i2)
for f in os.listdir("./imgs")[i1:i2]:
    filepath = os.path.join("./imgs", f)
    im = cv2.imread(filepath)
    imgs.append(im)
    names.append(filepath)


'''
code taken from https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
'''
config = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
config.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(config)
#predictor = build_model(config)


good_boxes = {}

cv2.namedWindow("pane")
cv2.setMouseCallback('pane',draw_circle)

i= 0
#while
last_index = [0]
last_img_id = None
while i < len(imgs):
    
    name = names[i]
    _, name = os.path.split(name)
    imgid = name[:3]
    #print('info')
    print(name)
    #print(imgid)
    #print(last_img_id)
    
    # if the image is from a new video, write the recorded boxes to a file and reload the data structure
    if last_img_id is None:
        last_img_id = imgid
    if last_img_id != imgid and len(good_boxes) != 0:
        #print('writing', last_img_id)
        write_to_json(good_boxes, last_img_id)
        last_img_id = imgid
        good_boxes = {}


    # if the image is from a video we've already seen, skip
    if 'boxes_{}.json'.format(imgid) in os.listdir(SAVEPATH):
        i += 1
        continue




    # perform inference
    im = imgs[i]
    #print(im.shape)
    height,width,c = im.shape
    outputs = predictor(im)

    # #print(outputs["instances"].pred_classes)
    # #print(outputs["instances"].pred_boxes)
    #print(names[i])


    gymnast_detected_flag =False
    visualize_img_flag = False
    
    # get the bounding box with the best criteria for containing the gymnast
    maxArea = -1
    maxIndex= -1

    maxScore = 0.0
    #print('average',np.average(im))
    if np.average(im) > 10:
        visualize_img_flag = True
    if len(outputs['instances'])>0:
        # find the largest box
        for j in range(len(outputs['instances'])):
            # if not human, disqualify
            if outputs['instances'][j].pred_classes != 0:
                continue
            box = outputs['instances'][j].pred_boxes
            score = outputs['instances'][j].scores
            
            # see if it's near center, disqualify if on edges of image
            centered, center_distance = near_center(box, height, width, score)
            if not centered:
                # pass
                continue
            
            # if box_area(box) > maxArea:# and score >= 0.7:
            #     maxArea = box_area(box)
            #     maxIndex=j

            # compute criteria and determine if it's the best
            # debug information
            tempScore = box_area(box) * score * score
            # #print(j)
            # #print(box_area(box))
            # #print(score)
            # #print(center_distance, 1-center_distance)
            # #print(tempScore)

            if tempScore > maxScore:
                maxScore = tempScore
                maxIndex=j
                # #print("max index set")
            #print()
        
        if maxIndex != -1:
            gymnast_detected_flag=True

    
    if visualize_img_flag:
        # We can use `Visualizer` to draw the predictions on the image.
        if gymnast_detected_flag:
            # visualize the best bbox
            predictions = outputs['instances'][maxIndex].to("cpu")
            
            #print(box_area(predictions.pred_boxes))
            box = predictions.pred_boxes
            key = render_and_wait(predictions, box)
            
        else:
            predictions = outputs['instances'].to("cpu")
            key = render_and_wait(predictions)
            print("defaulting w key")
            key = W_KEY



        # if space, good
        if key == SPACE_KEY:
            print("SPACE")
            print(predictions.pred_boxes)
            [[x1,y1,x2,y2]] = predictions.pred_boxes
            #print(x1,y1,x2,y2)
            #print(type(x1))
            x1 = max(x1-BOX_PADDING,torch.Tensor([0]))
            y1 = max(y1-BOX_PADDING, torch.Tensor([0]))
            x2 = min(x2+BOX_PADDING, torch.Tensor([width]))
            y2 = min(y2+BOX_PADDING, torch.Tensor([height]))
            #print(x1,y1,x2,y2)
            good_boxes[names[i]] = {"filename": names[i], "x1": x1.item(), "y1": y1.item(), "x2":x2.item(), "y2":y2.item()  }

        # if w, show more boxes
        # if w key, reshow the image with all the bounding boxes from inference to debug
        # if key == 119:
        if key == W_KEY:
            #print("W")
            predictions = outputs['instances'].to("cpu")
            key = render_and_wait(predictions)
            # if alt key, quit
            if key == ALT_KEY:
                write_to_json(good_bxes)
                quit()


            # r, click new box
            if key == R_KEY:
                #print("R")
                #print(mouseX, mouseY)
                pass
                # #print("here")
                # get mouse click location
                x,y = mouseX.pop(), mouseY.pop()
                # iterate over 0 class boxes until you find a box that contains the mouse click
                predictions = outputs['instances'].to("cpu")
                # #print("here")
                def click_in_box(x,y,box):
                    
                    # #print(box)
                    # #print(x,y)
                    x1,y1,x2,y2 = box
                    # quit()
                    if x1 <= x and x <= x2 and y1 <= y and y <= y2:
                        return True
                    return False
                for box in predictions.pred_boxes:
                    # #print("here")
                    if click_in_box(x,y,box):
                        x1,y1,x2,y2 = box
                        x1 = max(x1-BOX_PADDING,torch.Tensor([0]))
                        y1 = max(y1-BOX_PADDING, torch.Tensor([0]))
                        
                        x2 = min(x2+BOX_PADDING, torch.Tensor([width]))
                        y2 = min(y2+BOX_PADDING, torch.Tensor([height]))
                        key = render_and_wait(predictions, [[x1,y1,x2,y2]])
                        
                        good_boxes[names[i]] = {"filename": names[i], "x1": x1.item(), "y1": y1.item(), "x2":x2.item(), "y2":y2.item()}
                        break
                        # #print('appending', {"filename": names[i], "x1": x1.item(), "y1": y1.item(), "x2":x2.item(), "y2":y2.item()})

                # add this to good boxes
                # if a misclick, just move on to the next image and then hit backspace
        
            # d, draw new box
            if key == D_KEY:
                print("D")
                pass
                # get mouse click
                if len(mouseX) < 2 or len(mouseY) < 2:
                    pass
                    #print("error, no two boxes presented before d key pressed")
                else:
                    xm1,ym1 = mouseX.pop(), mouseY.pop()
                    # get second mouse click
                    xm2,ym2 = mouseX.pop(), mouseY.pop()

                    x1 = min(xm1, xm2)
                    x2 = max(xm1, xm2)
                    y1 = min(ym1, ym2)
                    y2 = max(ym1, ym2)

                    
                    x1 = max(x1-BOX_PADDING, 0)
                    y1 = max(y1-BOX_PADDING, 0)
                    x2 = min(x2+BOX_PADDING, width)
                    y2 = min(y2+BOX_PADDING, height)
                    # draw new box
                    key = render_and_wait(predictions, [[x1,y1,x2,y2]])
                    # discard key this time
                                    
                    good_boxes[names[i]] = {"filename": names[i], "x1": x1, "y1": y1, "x2":x2, "y2":y2}
                    # save information as x0 x1 y0 y1
                    # add this to good boxes
                    # if a misclick, just move on to the next image and then hit backspace
        
        if key == BACKSPACE_KEY:
            pass
            # skip the current image
            # and go back to the previous image
            # decrement index and continue
            i -= 3
            continue

        # if alt key, quit
        if key == ALT_KEY:
            write_to_json(good_boxes)
            quit()
        # if 0 num key, save as a good inference
        #print()
        last_index.append(i)
    gymnast_detected_flag=False
    i += 1
#end while

# save results
write_to_json(good_boxes)
