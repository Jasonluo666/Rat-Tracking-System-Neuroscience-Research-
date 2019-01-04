'''
    Copyright (C),2018, Shijie Zhao
    Filename: offline_kalman_filter.py
    Author: Shijie Zhao     Date: 12/10/2018
    Description:    Kalman filter is used to estimate the rat’s positions
                    when the measurement system fails to track the rat,
                    often in low light situations. The Kalman filter function
                    used is from Pykalman library.
    
    FunctionList:
        1. kalman_estimation
'''

from pykalman import KalmanFilter
import numpy as np

def kalman_estimation(measurements, tau = 1):
    '''
    Takes in as input:
    measurements: nx2 measurements array
    tau: delta t time duration for each frame

    return: kalman estimates of objects' positions and its state covariances

    First, the missing values at the beginning of the measurements array will be dropped.
    Then the measurements will be masked so that the kalman filter will know when there is a missing measurement and
    what to do with them.
    Then the kalman filter will initialize and expectation-maximization function will approximate the error covariances.
    Finally the kalman filter will make estimations of position of the object and return the result:
    kalman_estimates, kalman_filtered_state_covariances
    '''
    lenMeasurements = len(measurements)

    # kalman_predictions should have the same length as number of frames, empty 
    # prediction is [-1, -1]
    kalman_predictions = np.ones((lenMeasurements, 4))*-1

    # in measurements: missing measurements are stored as [-1, -1]
    # remove missing measurements at the beginning, when no object is within frame
    while True:
        if measurements[0,0]==-1.:
            measurements=np.delete(measurements,0,0)
        else:
            break
    # mask measurements so filter knows what to do with them
    # mask values less than 0
    # MarkedMeasure does not contain missing values in the begining
    MarkedMeasure=np.ma.masked_less(measurements,0)

    initX = MarkedMeasure[0,0]
    initY = MarkedMeasure[0,1]
    initVx = MarkedMeasure[1,0] - MarkedMeasure[0,0]
    initVy = MarkedMeasure[1,1] - MarkedMeasure[0,1]
    initstate = [initX, initY, initVx, initVy]

    #transition mat
    A = [[1, 0, tau, 0], [0, 1, 0, tau], [0, 0, tau, 0], [0, 0, 0, tau]]
    observation_matrix = [[1, 0,0,0], [0,1,0,0]]

    # observation matrix is H matrix, or the transition matrix between kalman 
    # estimate and it's position
    kf = KalmanFilter(
                      transition_matrices=A,
                      observation_matrices=observation_matrix,
                      initial_state_mean=initstate,
                      em_vars=['transition_covariance', 'observation_covariance', 'initial_state_covariance']
    )

    # expectation_maximization parameter search
    kf.em(MarkedMeasure, n_iter=5)

    # kalman filter estimate
    (filtered_state_means, filtered_state_covariances) = kf.filter(MarkedMeasure)

    # matrix used to store result
    # put estimate into new matrix including missing measurements at the begining
    # that dont have an estimate, so the result can be mapped to each frame
    # empty estimate [-1, -1, -1, -1] is caused by the missing measurements at the beginning
    firstEstimateIdx = lenMeasurements - len(filtered_state_means)
    kalman_estimates = np.ones((lenMeasurements, 4))* -1
    kalman_filtered_state_covariances = np.ones((lenMeasurements, 4, 4))* -1

    kalman_estimates[firstEstimateIdx:] = filtered_state_means
    # covarince matrix also mapped to have same length as measurements
    kalman_filtered_state_covariances[firstEstimateIdx:] = filtered_state_covariances

    return kalman_estimates, kalman_filtered_state_covariances