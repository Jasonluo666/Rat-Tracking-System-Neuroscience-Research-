'''
    Copyright (C),2018, Shijie Zhao
    Filename: visualization.py
    Author: Shijie Zhao     Date: 12/10/2018
    Description:    Visualize the head/body positions and velocity of rat, and generate a video.
    
    FunctionList:  
        1. visualize_tracking
    
    To Do Function:
        1. draw_trace
'''

import cv2
import numpy as np

VECTOR_RESCALER = 10

def visualize_tracking(video, kalman_estimates, measurements, body_measurements):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    numframes=cap.get(7)

    video_name = video.split('/')[-1]
    video_name = video_name.split('.')[0]

    out = cv2.VideoWriter('../output/' + video_name + '_output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(fps), (int(frame_width), int(frame_height)))
    
    isEmpty, frame = cap.read()
    frame_id = 0
    while isEmpty:
        present_estimation = np.array(kalman_estimates[frame_id])#.astype(np.int)
        orientationVec = (present_estimation[2:]*VECTOR_RESCALER).astype(np.int)

        present_estimation = present_estimation.astype(np.int)
        
        # head kalman estimation
        cv2.drawMarker(frame, (present_estimation[0], present_estimation[1]), (255, 0,0),  cv2.MARKER_CROSS, 5, 1)
        
        # orientation
        cv2.line(frame, tuple(present_estimation[0:2]), tuple((present_estimation[0:2]+orientationVec)), (0, 0, 255), 1)
        
        # body
        cv2.drawMarker(frame, tuple(body_measurements[frame_id]), (0, 255, 0),  cv2.MARKER_TILTED_CROSS, 5, 1)
        
        cv2.imshow('frame',frame)
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        isEmpty, frame = cap.read()

        frame_id = frame_id + 1

def draw_trace(frame_id):
	pass