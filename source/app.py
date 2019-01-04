'''
    Copyright (C),2018, Jiangshan Luo, Shijie Zhao
    Filename: app.py
    Author: Jiangshan Luo, Shijie Zhao     Date: 12/10/2018
    Description:    Main function of the program
    
    FunctionList:  
        1.main function
'''

from UI import *
import segmentation
from offline_kalman_filter import *
from visualization import *
import threading
import scipy.io as sio

# new thread -> run the configuration tasks
class MyThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
    def run(self):
        self.func(self.args[0], self.args[1])

def main():
    config_window = config()

    # choose videos
    video= config_window.choose_video()

    # select the region for the experiment
    region = segmentation.select_region(video)
    print('select region:', region)

    # threshold configuration
    thread = MyThread(segmentation.config_segmentation, [video, region])
    thread.start()

    segmentation.stop_signal = False
    config_window.run_config()
    segmentation.stop_signal = True

    threshold = int(config_window.get_params())

    if config_window.get_params() is not None:
        # get the localization of rat's head and body, return lists
        head_loc_history, body_loc_history, fps = segmentation.segmentation(video, threshold, region)

        # pass the measurement data to Kalman Filter -> position estimation and velocity prediction
        measurements = np.array(head_loc_history)
        kalman_estimates, filtered_state_covariances = kalman_estimation(measurements)
        prediction = kalman_estimates
        
        # save prediction to .mat
        video_name = video.split('/')[-1]
        video_name = video_name.split('.')[0]
        sio.savemat('../data/' + video_name + '_data', {'kalman_estimates': kalman_estimates, 'measurements': measurements, 'filtered_state_covariances': filtered_state_covariances})

        # visualize the head/body positions and velocity
        visualize_tracking(video, kalman_estimates, measurements, body_loc_history)
    
if __name__ == "__main__":
    main()