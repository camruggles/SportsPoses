import os
import sys
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import cv2
import json
import numpy as np

from test_config import cfg
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from utils.imutils import im_to_numpy, im_to_torch
from networks import network 
from dataloader.mscocoMulti import MscocoMulti
from tqdm import tqdm
import pdb

def main(args):
    # create model
    model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, pretrained = False)
    model = torch.nn.DataParallel(model).cuda()

    test_loader = torch.utils.data.DataLoader(
        MscocoMulti(cfg, train=False),
        batch_size=args.batch*args.num_gpus, shuffle=False,
        num_workers=args.workers, pin_memory=True) 

    # load trainning weights
    checkpoint_file = os.path.join(args.checkpoint, args.test+'.pth.tar')
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))
    
    # change to evaluation mode
    model.eval()
    
    print('testing...')
    full_result = []

    '''
    inputs : 16x3x388x288 tensor for the images
    meta: dictionary comprised of
    dict_keys(['index', 'imgID', 'GT_bbox', 'img_path', 'augmentation_details', 'det_scores'])
    not sure what aug details are, det scores, or where the keypoints gt are and where gt bbox comes from
    '''

    '''
    does the code only grab the pixels for the body and resize them, and then compare to the gt outside the code
    '''
    for i, (inputs, meta) in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs.cuda())
            if args.flip == True:
                flip_inputs = inputs.clone()
                for i, finp in enumerate(flip_inputs):
                    finp = im_to_numpy(finp)
                    finp = cv2.flip(finp, 1)
                    flip_inputs[i] = im_to_torch(finp)
                flip_input_var = torch.autograd.Variable(flip_inputs.cuda())

            # compute output
            global_outputs, refine_output = model(input_var)
            score_map = refine_output.data.cpu()
            score_map = score_map.numpy()
            # score map?

            if args.flip == True:
                flip_global_outputs, flip_output = model(flip_input_var)
                flip_score_map = flip_output.data.cpu()
                flip_score_map = flip_score_map.numpy()

                for i, fscore in enumerate(flip_score_map):
                    fscore = fscore.transpose((1,2,0))
                    fscore = cv2.flip(fscore, 1)
                    fscore = list(fscore.transpose((2,0,1)))
                    for (q, w) in cfg.symmetry:
                       fscore[q], fscore[w] = fscore[w], fscore[q] 
                    fscore = np.array(fscore)
                    score_map[i] += fscore
                    score_map[i] /= 2

            ids = meta['imgID'].numpy()
            det_scores = meta['det_scores']
            # wtf is happening in this code??
            for b in range(inputs.size(0)):
                details = meta['augmentation_details'] #''' why are we augmenting the evaluation set? '''
                single_result_dict = {}
                single_result = []
                
                single_map = score_map[b]
                r0 = single_map.copy()
                r0 /= 255
                r0 += 0.5
                v_score = np.zeros(17) # visibility score?
                for p in range(17): 
                    # they average the flipped score with the real score
                    single_map[p] /= np.amax(single_map[p])
                    border = 10
                    dr = np.zeros((cfg.output_shape[0] + 2*border, cfg.output_shape[1]+2*border)) # image shape with a 10 border
                    dr[border:-border, border:-border] = single_map[p].copy() # copy image into border
                    dr = cv2.GaussianBlur(dr, (21, 21), 0) # described in paper
                    lb = dr.argmax() # 
                    y, x = np.unravel_index(lb, dr.shape) # 
                    dr[y, x] = 0
                    lb = dr.argmax()
                    py, px = np.unravel_index(lb, dr.shape) # getting a nother peak?
                    y -= border
                    x -= border
                    py -= border + y
                    px -= border + x
                    ln = (px ** 2 + py ** 2) ** 0.5 # no clue
                    delta = 0.25

                    # a quarter offset in the direction from the highest to second highest response is used
                    if ln > 1e-3:
                        x += delta * px / ln
                        y += delta * py / ln
                    x = max(0, min(x, cfg.output_shape[1] - 1))
                    y = max(0, min(y, cfg.output_shape[0] - 1))
                    # de augmentation
                    resy = float((4 * y + 2) / cfg.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                    resx = float((4 * x + 2) / cfg.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                    # rescoring strategy occurs somewhere, product of box score and average score
                    v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])
                    single_result.append(resx)
                    single_result.append(resy)
                    single_result.append(1)
                if len(single_result) != 0:
                    single_result_dict['image_id'] = int(ids[b])
                    single_result_dict['category_id'] = 1
                    single_result_dict['keypoints'] = single_result # single_result contains 17x3 predictions, but pixels are out of bounds of 384x288
                    single_result_dict['score'] = float(det_scores[b])*v_score.mean() # rescoring happens here
                    
                    # load the original image
                    filename = meta['img_path'][b]
                    img = cv2.imread(filename)
                    # plot dots on the pixels described from single_result_dict
                    keypoints= single_result_dict['keypoints']
                    #print(img.shape)
                    for x,y,v,p in zip(keypoints[0::3], keypoints[1::3], keypoints[2::3], range(17)):
                        if v != 1:
                            print(filename)
                        print(x,y, v, v_score[p])
                        img = cv2.circle(img, (int(x),int(y)), radius=3, color=(0, 0, 255), thickness=-1)
                        #print(img.shape)
                    # save to a file
                    new_filename = './plots/dotted_'+os.path.split(filename)[-1][:-4]+"_"+str(b)+".jpg"
                    cv2.imwrite(new_filename, img)
                    print(new_filename)
                    #print(v_score)
                    full_result.append(single_result_dict)
        break

    quit()
    result_path = args.result
    # write all the results to a json
    if not isdir(result_path):
        mkdir_p(result_path)
    result_file = os.path.join(result_path, 'result.json')
    with open(result_file,'w') as wf:
        json.dump(full_result, wf)

    # evaluate on COCO
    # reload the json just saved, and the gt json and call the coco api
    eval_gt = COCO(cfg.ori_gt_path)
    eval_dt = eval_gt.loadRes(result_file)
    ''' keypoints come from COCO eval func call?? '''
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPN Test')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')      
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to load checkpoint (default: checkpoint)')
    parser.add_argument('-f', '--flip', default=True, type=bool,
                        help='flip input image during test (default: True)')
    parser.add_argument('-b', '--batch', default=128, type=int,
                        help='test batch size (default: 128)')
    parser.add_argument('-t', '--test', default='CPN384x288', type=str,
                        help='using which checkpoint to be tested (default: CPN256x192')
    parser.add_argument('-r', '--result', default='result', type=str,
                        help='path to save save result (default: result)')
    main(parser.parse_args())
