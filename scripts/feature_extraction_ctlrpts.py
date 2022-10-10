# feature extraction control points
import numpy as np 
import cv2
from numpy import linalg as LA
from geomdl import BSpline
import pickle
import sys
import pandas as pd


def rect2pol(coord):
    rho = np.sqrt(coord[0]**2 + coord[1]**2)
    phi = np.arctan2(coord[1], coord[0])
    return np.array([rho, phi])

def pol2rect(coord):
    x = coord[0] * np.cos(coord[1])
    y = coord[0] * np.sin(coord[1])
    return np.array([x, y])

# estimating image segmentations between the two segmented frames--end of diastole and end of systole
def feature_extract(control_points):

    feature_vector = []

    control_points = np.array(control_points)
    # control_points[0] are the ctlpts at the end of diastole
    # control_points[1] are the ctlpts at the end of systole
    # control points 0 and 4 define main axis in each of the two segmented images
    main_axis_ini = control_points[0][4] - control_points[0][0]
    main_axis_end = control_points[1][4] - control_points[1][0]

    # normalization factor w
    factor = 5.0
    w = np.linalg.norm(main_axis_ini) / factor
    print('normalization factor: ', w)
    # axis normalization

    main_axis_ini = main_axis_ini / np.linalg.norm(main_axis_ini)
    main_axis_end = main_axis_end / np.linalg.norm(main_axis_end)
    
    # print('main axes: ')
    # print(main_axis_ini)
    # print(main_axis_end)

    # defining a new main axis that depends of the number of frames between the segmented images
    # first coordinates transformation rect to polar 
    
    # diff_main_axis = main_axis_end - main_axis_ini
    diff_main_axis = rect2pol(main_axis_end) - rect2pol(main_axis_ini)
    
    # delta_main_axis = diff_main_axis / (len(frames)-1)
    print("diff_main_axis")
    print(diff_main_axis)
    # print(delta_main_axis)

    # first feature: angle difference between central axes of end of systole and end of diastole
    feature_vector.append(diff_main_axis[1])

    # the origin (apex location) is also changing from frame to frame
    # defining origin for each frame (new_origin)
    diff_origins = control_points[1][0] - control_points[0][0]
    # delta_origins = diff_origins / (len(frames)-1)

    print('diff_origins: ', diff_origins)
    print('diff_origins/w: ', diff_origins/w)

    # second feature: origin difference normalized by w
    feature_vector.append((diff_origins/w).tolist())


    # control points are also changing from frame to frame
    # calculating the coordinates of each control point on its respective frame of reference
    # first: subtract origin for the respective first and end frames
    pts_ini = control_points[0] - control_points[0][0]
    pts_end = control_points[1] - control_points[1][0]
    # print(control_points[0])
    # print(pts_ini) 
    #second: calculating components of each resultant vector in the main_axis frame of reference
    # calculating secundary axis that is perpendicular to the main axis for the initial and end frames
    secu_axis_ini = np.cross(main_axis_ini, (0,0,-1))[0:2]
    secu_axis_end = np.cross(main_axis_end, (0,0,-1))[0:2]

    xs_ini = np.dot(pts_ini, secu_axis_ini)
    ys_ini = np.dot(pts_ini, main_axis_ini)

    xs_end = np.dot(pts_end, secu_axis_end)
    ys_end = np.dot(pts_end, main_axis_end)

    # third: calculating deltas between two consecutive frames
    diff_xs = xs_end - xs_ini
    diff_ys = ys_end - ys_ini

    print('diff_xs')
    print(diff_xs)
    print(diff_xs/w)
    print('diff_ys')
    print(diff_ys)
    print(diff_ys/w)

    # third feature: distance differences between control points, x vector and y vector
    feature_vector.append((diff_xs/w).tolist())
    feature_vector.append((diff_ys/w).tolist())


    return feature_vector 


####### main function ###########
if __name__== '__main__':

    print('feature extraction from control points')
    # read control points associated with two frames: End of Diastole and End of Systole
    filename_controlpts = 'data/controlpts_segmentations'

    # reading list of segmented sequences
    with open(filename_controlpts, 'rb') as fp:
        # the frames count started at 1 (first frame)
        print('reading data ...')
        [file_name_list, control_points_list] = pickle.load(fp)
        print ('done.')
    # print('filenames: ', len(file_name_list), file_name_list)
    # print('filenames: ', len(control_points_list), control_points_list[0])

    features_cpts = feature_extract(control_points_list[0])
    print('feature vector:')
    print(features_cpts)



