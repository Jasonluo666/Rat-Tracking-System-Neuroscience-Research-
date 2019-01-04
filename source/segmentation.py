'''
    Copyright (C),2018, Jiangshan Luo
    Filename: segmentation.py
    Author: Jiangshan Luo     Date: 12/10/2018
    Description:    Segmentation of the rat's head and body using 
                    thresholding and morphological transformations.
    
    FunctionList:  
        1. get_videos
        2. find_head_pos
        3. find_body_pos
        4. adaptiveThresh
        5. absoluteThresh
        6. segmentation
        7. config_segmentation
        8. select_region
'''

# data loading
import cv2
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, abspath, exists
import random
import scipy.io as sio

# global variables
random.seed = 1
kernel = np.ones((2,2),np.uint8)
kernel_x = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
head_dist_threshold = 20
body_dist_threshold_max = 15
dist_threshold = 50
circularity_threshold = 0.5
block_head_area = 7
area_max = 20
not_found_flag = [-1, -1]
move_length_thresh = 3
stop_signal = False
# -------------------------------------------------------------------

# read video path from the directory
def get_videos(img_dir):
    # read data
    video_path = abspath(img_dir)
    video_list = [join(video_path, file) for file in listdir(video_path) if isfile(join(video_path, file)) and 'mp4' in file] 
    video_list.sort()
    
    return video_list

# find the position of rat's head
def find_head_pos(absolute_thresh_img, frame, head_loc_history):
    _, contours, hierarchy = cv2.findContours(absolute_thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # calculate the circularity of blobs
    circularity = []
    for index in range(len(contours)):
        (x,y),radius = cv2.minEnclosingCircle(contours[index])
        center = (int(x),int(y))
        radius = int(radius)
        
        area = cv2.contourArea(contours[index])
        circle_area = np.math.pi * (radius ** 2)

        # filter out the blobs with high circularity values
        if circle_area > 0 and area / circle_area > circularity_threshold:
            circularity.append((area / circle_area, index, center))

    head_pos = None

    # filter the contours
    if len(circularity) >= 2:
        min_dist = np.inf
        min_loc = None

        # choose the closest points (two LEDs)
        for index_1 in range(len(circularity)):
            for index_2 in range(index_1 + 1, len(circularity)):
                dist = np.sqrt(np.sum((np.array(circularity[index_1][2]) - np.array(circularity[index_2][2])) ** 2))
                if dist < min_dist:
                    min_dist = dist
                    min_loc = (index_1, index_2)
        if min_loc is not None and min_dist < head_dist_threshold:
            head_pos = ((np.array(circularity[min_loc[0]][2]) + np.array(circularity[min_loc[1]][2])) / 2).astype(int).tolist()
    
    # if not found, find the closest one blob
    if len(head_loc_history) > 0 and head_loc_history[-1] != not_found_flag:
        last_loc = head_loc_history[-1]
        if head_pos is None:
            min_dist = np.inf
            min_centroid = None
            for contour in contours:
                M = cv2.moments(contour)
                cx = int(M['m10']/(M['m00'] + 0.0001))
                cy = int(M['m01']/(M['m00'] + 0.0001))

                dist = np.sqrt((cx - last_loc[0]) ** 2 + (cy - last_loc[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    min_centroid = [cx, cy]
            
            if min_dist < dist_threshold:
                head_pos = min_centroid
        else:
            dist = np.sqrt((head_pos[0] - last_loc[0]) ** 2 + (head_pos[1] - last_loc[1]) ** 2)
            
            if dist > dist_threshold:
                head_pos = None
    
    # mark if the rat's head is not found
    if head_pos is None:
        head_loc_history.append(not_found_flag)
    else:
        head_loc_history.append(head_pos)
    
    if head_pos is not None:
        cv2.circle(frame, tuple(head_pos), 3, (0,255,0), -1)

# find the position of rat's body
def find_body_pos(adaptive_thresh_img, frame, head_loc_history, body_loc_history):
    body_loc = None

    # find the contours of detected blobs
    _, contours, hierarchy = cv2.findContours(adaptive_thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if head_loc_history[-1] != not_found_flag:
        head_pos = np.array(head_loc_history[-1])

        # if the rat's head location is found, use it to find the direction of movement
        if len(head_loc_history) > 1 and head_loc_history[-2] != not_found_flag:
            previous_head_pos = np.array(head_loc_history[-2])
            direction = head_pos - previous_head_pos
            move_length = np.sqrt(np.sum(np.square(direction)))
        else:
            direction = None
        
        max_area = 0
        for contour in contours:
            M = cv2.moments(contour)
            cx = int(M['m10']/(M['m00'] + 0.0001))
            cy = int(M['m01']/(M['m00'] + 0.0001))

            dist2head = np.sqrt((cx - head_pos[0]) ** 2 + (cy - head_pos[1]) ** 2)
            area = cv2.contourArea(contour)
            
            if direction is not None and move_length > move_length_thresh:
                estimated_direction = head_pos - np.array([cx, cy])
                is_same_direction = (direction[0] * estimated_direction[0] > 0) and (direction[1] * estimated_direction[1] > 0)

                if dist2head < body_dist_threshold_max and area > max_area and is_same_direction:
                    max_area = area
                    body_loc = [cx, cy]
            elif dist2head < body_dist_threshold_max and area > max_area:
                max_area = area
                body_loc = [cx, cy]
    
    # mark if the rat's body is not found
    if body_loc is None:
        body_loc_history.append(not_found_flag)
    else:
        body_loc_history.append(body_loc)
    
    if body_loc is not None:
        # biased_body_loc = (2 * body_loc[0] - head_pos[0], 2 * body_loc[1] - head_pos[1])
        cv2.circle(frame, tuple(body_loc), 3, (255,0,0), -1)

# adaptive thresholding
def adaptiveThresh(img_gray, region):
    img_gray = cv2.medianBlur(img_gray,5)

    adaptive_thresh_img = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 19)
    adaptive_thresh_img = cv2.erode(adaptive_thresh_img, kernel_x, iterations=1)
    adaptive_thresh_img = cv2.dilate(adaptive_thresh_img, kernel_x, iterations=2)
    
    area_of_interest = adaptive_thresh_img
    mask = np.zeros_like(area_of_interest)
    cv2.fillConvexPoly(mask, np.array(region), 255)
    area_of_interest[mask < 255] = 0

    return adaptive_thresh_img

# absolute thresholding
def absoluteThresh(img_gray, min_thresh, region):
    area_of_interest = img_gray.copy()
    mask = np.zeros_like(area_of_interest)
    cv2.fillConvexPoly(mask, np.array(region), 255)
    area_of_interest[mask < 255] = 0

    _ ,absolute_thresh_img = cv2.threshold(area_of_interest, min_thresh, 255, cv2.THRESH_BINARY)
    absolute_thresh_img = cv2.erode(absolute_thresh_img, kernel, iterations=2)
    absolute_thresh_img = cv2.dilate(absolute_thresh_img, kernel, iterations=2)

    mask = np.zeros_like(absolute_thresh_img)
    cv2.fillConvexPoly(mask, np.array(region), 255)
    absolute_thresh_img[mask < 255] = 0

    return absolute_thresh_img

# main function of segmentation, generate the locations of rat's head and body
def segmentation(video, threshold, region):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    isEmpty, frame = cap.read()
    
    frame_id = 0
    head_loc_history = []
    body_loc_history = []

    # read the video frames
    while isEmpty:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # thresholding
        absolute_thresh_img = absoluteThresh(img_gray, threshold, region)
        adaptive_thresh_img = adaptiveThresh(img_gray, region)
        
        # find the led lights in the images
        find_head_pos(absolute_thresh_img, frame, head_loc_history)
        find_body_pos(adaptive_thresh_img, frame, head_loc_history, body_loc_history)

        # show the images
        cv2.imshow('absolute_thresh_img',absolute_thresh_img)
        cv2.imshow('adaptive_thresh_img',adaptive_thresh_img)
        cv2.imshow('frame',frame)
        frame_id = frame_id + 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        isEmpty, frame = cap.read()
    
    cv2.destroyAllWindows()

    # compute the accuarcy of finding rat's head and body
    percent_head = np.sum(np.array(head_loc_history) != not_found_flag) / (2 * frame_id)
    percent_body = np.sum(np.array(body_loc_history) != not_found_flag) / (2 * frame_id)
    print('head found percent:', percent_head)
    print('body found percent:', percent_body)
    
    return head_loc_history, body_loc_history, fps


# segmentation for finding the threshold (configuration)
def config_segmentation(video, region):
    cap = cv2.VideoCapture(video)
    isEmpty, frame = cap.read()
    print('---',np.std(frame))
    def nothing(x):
        pass
    cv2.namedWindow('result')
    cv2.createTrackbar('Colorbars', 'result', 0, 255, nothing)

    loc_history = []
    while isEmpty and stop_signal == False:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        min_thresh = cv2.getTrackbarPos("Colorbars", "result")
        absolute_thresh_img = absoluteThresh(img_gray, min_thresh, region)
        
        # find the led lights in the images
        find_head_pos(absolute_thresh_img, frame, loc_history)

        cv2.imshow('config_thresh_img',absolute_thresh_img)
        cv2.imshow('config_frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        isEmpty, frame = cap.read()
    
    cv2.destroyAllWindows()

# GUI to find the area of interest (configuration)
def select_region(video):
    cap = cv2.VideoCapture(video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    isEmpty, frame = cap.read()
    
    count = 0
    while count < length / 10:
        isEmpty, frame = cap.read()
        count += 1

    # mouse callback function
    def draw(event,x,y,flags,param):
        global img, vertex
        
        if event == cv2.EVENT_LBUTTONDOWN:
            vertex.append([x, y])
            
            pts = np.array(vertex, np.int32)
            cv2.fillConvexPoly(img, pts, (0,255,0))
        elif event == cv2.EVENT_RBUTTONDOWN:
            vertex.pop()
            
            if len(vertex) > 0:
                img = frame.copy()
                pts = np.array(vertex, np.int32)
                cv2.fillConvexPoly(img, pts, (0,255,0))
    
    global img, vertex
    img = frame.copy()
    vertex = []
    # Create a black image, a window and bind the function to window
    cv2.namedWindow('select region')
    cv2.setMouseCallback('select region',draw)

    while len(vertex) < 5:
        cv2.imshow('select region',img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    return vertex[:4]
