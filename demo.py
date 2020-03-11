#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from datetime import timedelta
from darknet import darknet
#from yolo import YOLO
import csv
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
#from deep_sort.territories import P2
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from misc.color import get_rgb
from misc.utils import blend_transparent, area_inside
from shapely.geometry import Polygon, box
warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
netMain = None
metaMain = None
altNames =['AdultBird', 'Chick','Egg']
dw=1920.0/416.0
dh=1080.0/416.0

def  convert(detections,dw,dh ):
    boxs=[]
    class_ids=[]
    for detection in detections:
        
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        boxs.append([(x-w/2)*dw,(y-h/2)*dh,w*dw,h*dh])
        class_ids.append(altNames.index(detection[0].decode("utf-8") ))
    return (boxs, class_ids)


def main():

    global metaMain, netMain, altNames
    configPath = "darknet/cfg/yolov3-seabird.cfg"
    weightPath = "darknet/backup_608/yolov3-seabird_final.weights"
    metaPath = "darknet/cfg/seabird.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass


   # Definition of the parameters
    max_cosine_distance = 0.3
    max_euclidean_distance = 150.0
    nn_budget = None
    nms_max_overlap = 1
   # deep_sort 
    
    metric = nn_matching.NearestNeighborDistanceMetric("euclidean", max_euclidean_distance, nn_budget)
    #metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = False
    
    #video_capture = cv2.VideoCapture('/data/Farallon3_20190706_000001_No001.avi')
    #video_capture = cv2.VideoCapture('/data/15_fps/Farallon3_20190620_021546_No001.mp4')
    video_capture = cv2.VideoCapture('/data/15_fps/Farallon3_20190621_090300_No004.mp4')
    #video_capture = cv2.VideoCapture('/data/rows_data/15_fps/Farallon3_20190603_155007_No001.avi')

    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    #video_fps = 25
    list_file = open('events.csv', 'w')
    track_log_file = open('track_log.csv', 'w')
    wr = csv.writer(list_file, dialect='excel')
    wr_tracklog = csv.writer(track_log_file, dialect='excel')
    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('1.avi', fourcc, 15, (w, h))
        zone = cv2.imread('mask/test_zone.png', -1)
     # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    
    fps = 0.0
    frame_index = -1 
    mask = cv2.imread('mask/mask_new.jpg')
    mask = np.uint8(mask/255)
    ##############################################
    #points = [(130,102),(535,370),(808,345),(1570,391),(1494,808),(373,817),(4,496),(1,276)]
    #points = [(4,22),(121,96),(207,99),(537,366),(819,324),(1564,385),(1764,322),(1648,889),(105,915),(3,762)]
    points = [(2,24),(127,24),(579,321),(1746,336),(1674,878),(1912,957),(1926,1081),(2,1074)]
    zone_polygon = Polygon(points)
    text_count_size = 9e-3 * 200
    text_count_x,text_count_y = 550,  1000
    avg_area_box_all_frames = 0
    time_stamp = datetime(2019, 6, 21, 10, 33, 5) 
    
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        frame_index = frame_index +1  
        t1 = time.time()
        write = 0 
        if frame_index % 15 == 0:
           write = 1 
            
        draw_frame  = frame 
        frame = np.multiply(frame, mask)
        # image = Image.fromarray(frame)
        #image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.5)

        boxs, class_ids=convert(detections, dw, dh)
        
        # score to 1.0 here).
        dets = [Detection(bbox, 1.0, class_id) for bbox,class_id in zip(boxs, class_ids)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in dets])
        scores = np.array([d.confidence for d in dets])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections_tracker = [dets[i] for i in indices]
        ids = [class_ids[i] for i in indices]
        time_lapsed_sec = frame_index / video_fps
        time_stamp_now = time_stamp + timedelta(seconds=time_lapsed_sec) 

        # Call the tracker
        tracker.predict()
        tracker.update(detections_tracker, ids, time_stamp_now)
        tracker.update_events(draw_frame, time_stamp_now, wr, wr_tracklog, write )
   
        avg_area_box = 0
        for det in detections:
            name , x, y, w, h = det[0],det[2][0],det[2][1],det[2][2],det[2][3]
            class_id = altNames.index(name.decode("utf-8") )
            bbox = [(x-w/2)*dw,(y-h/2)*dh,(x+w/2)*dw,(y+h/2)*dh]
            area_box = w*h*dw*dh
            if area_box > 3*avg_area_box_all_frames:
                 cv2.rectangle(draw_frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 3)
                 cv2.putText(draw_frame,"Flapping detected !!!!",(text_count_x,text_count_y),4,text_count_size, (255,0,0),4) 
            avg_area_box = avg_area_box + area_box
            cv2.rectangle(draw_frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),get_rgb(class_id,3), 2)
        avg_area_box = avg_area_box/len(detections)
        avg_area_box_all_frames=avg_area_box 
        
        if writeVideo_flag:
            # save a frame
            out.write(draw_frame)
      
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
