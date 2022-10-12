# feature extraction intensities along the curve
import numpy as np
import cv2
import pickle
import pandas as pd
from geomdl import BSpline
import matplotlib.pyplot as plt

def rect2pol(coord):
    rho = np.sqrt(coord[0]**2 + coord[1]**2)
    phi = np.arctan2(coord[1], coord[0])
    return np.array([rho, phi])

def pol2rect(coord):
    x = coord[0] * np.cos(coord[1])
    y = coord[0] * np.sin(coord[1])
    return np.array([x, y])


#### function read video ultrasound ####
def read_frames(dir_videos, data_coord, ini_cont_rows):

    selected_frames = []
    # first image
    cont_rows=ini_cont_rows
    # file_name and new_file_name start with same values but they could change during the program
    file_name = data_coord.at[cont_rows,'FileName']
    new_file_name = data_coord.at[cont_rows,'FileName']
    # frame_number and new_frame_number start with same values but they could change during the program
    frame_number = data_coord.at[cont_rows, 'Frame']
    new_frame_number = data_coord.at[cont_rows, 'Frame']
    print('first: ',file_name, frame_number)

    # initial frame id
    id_frame_ini = frame_number

    cap = cv2.VideoCapture(dir_videos+file_name)

    segmented_frames = 0
    save_frames = False
    cont_frames=0

    while cap.isOpened() and segmented_frames < 2:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if cont_frames == frame_number:

            img = np.copy(frame)

            # drawing image segmentation--lines in the left ventricle--on a selected frame (end of diastole or end of systole)
            while file_name == new_file_name and frame_number == new_frame_number:

                x1 = round(data_coord.at[cont_rows,'X1'])
                y1 = round(data_coord.at[cont_rows, 'Y1'])

                x2 = round(data_coord.at[cont_rows,'X2'])
                y2 = round(data_coord.at[cont_rows, 'Y2'])

                # drawing image segmentation, line by line
                cv2.line(img,(x1,y1),(x2,y2),(255,255,0),1)

                cont_rows+=1
                new_file_name = data_coord.at[cont_rows,'FileName']
                new_frame_number = data_coord.at[cont_rows,'Frame']

            # resizing for a better image visualization
            img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
            img = cv2.resize(img,(448,448),interpolation=cv2.INTER_CUBIC)

            # saving images with their respective segmentations
            if save_frames == False:
                # saving first segmented frame
                frame_ini = np.copy(img)
                selected_frames.append(frame)
                save_frames = True
                # print('segmented image: ',file_name, frame_number)
                # print('frame number: ', cont_frames)
            else:
                # saving second (last) segmented frame
                frame_end = np.copy(img)
                selected_frames.append(frame)
                save_frames = False
                # print('segmented image: ',file_name, frame_number)
                # print('frame number: ', cont_frames)

            frame_number = new_frame_number
            segmented_frames+=1

        elif save_frames == True:
            # saving frames in between the two segmented images
            selected_frames.append(frame)

        else:
            pass

        cont_frames+=1

    return file_name, selected_frames, frame_ini, frame_end, cont_rows


# estimating image segmentations between the two segmented frames--end of diastole and end of systole
def segmentation_interpolation(control_points, frames):

    control_points = np.array(control_points)
    # control points 0 and 4 define main axis in each of the two segmented images
    main_axis_ini = control_points[0][4] - control_points[0][0]
    main_axis_end = control_points[1][4] - control_points[1][0]

    # axis normalization

    main_axis_ini = main_axis_ini / np.linalg.norm(main_axis_ini)
    main_axis_end = main_axis_end / np.linalg.norm(main_axis_end)

    # defining a new main axis that depends of the number of frames between the segmented images

    # diff_main_axis = main_axis_end - main_axis_ini
    diff_main_axis = rect2pol(main_axis_end) - rect2pol(main_axis_ini)

    # the origin (apex location) is also changing from frame to frame
    # defining origin for each frame (new_origin)
    diff_origins = control_points[1][0] - control_points[0][0]

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
    
    # calcualting ratio of increments amount frames using a cosine function
    alphas = np.linspace(0, 3.14159265, len(frames))
    ratios = (1-np.cos(alphas)) / 2.0

    frames_control_pts_1 = []
    main_axis_ini_polar = rect2pol(main_axis_ini)
    # the main axis is changing from frame to frame
    for frame_number in np.arange(len(frames)):

        new_main_axis_1 = main_axis_ini_polar + diff_main_axis*ratios[frame_number]
        new_main_axis_1 = pol2rect(new_main_axis_1)
        # print('y_new:', new_main_axis)

        # main axis is y axis, new_secu_axis is x axis, and (0,0,-1) is z axis
        # applying cross product and taking only two components of the result
        new_secu_axis_1 = np.cross(new_main_axis_1, (0,0,-1))[0:2]
        # print('x_new:', new_secu_axis)

        # the origin (apex location) is also changing from frame to frame
        # defining origin for each frame (new_origin)
        new_origin_1 = control_points[0][0] + diff_origins*ratios[frame_number]

        # fourth: adding deltas according to the frame number
        new_xs_1 = xs_ini + diff_xs*ratios[frame_number]
        new_ys_1 = ys_ini + diff_ys*ratios[frame_number]


        # reshaping to apply matrix multiplication
        # assembling all to calculate coordinates in the original frame of reference
        new_main_axis_1 = np.reshape(new_main_axis_1, (1,2))
        new_secu_axis_1 = np.reshape(new_secu_axis_1, (1,2))

        new_xs_1 = np.reshape(new_xs_1, (-1,1))
        new_ys_1 = np.reshape(new_ys_1, (-1,1))

        new_pts_1 = new_xs_1*new_secu_axis_1 + new_ys_1*new_main_axis_1 + new_origin_1

        # frames_control_pts_0.append(new_pts_0.tolist())
        frames_control_pts_1.append(new_pts_1.tolist())


    return frames_control_pts_1


# drawing image segmentations on all selected frames of each video
def drawing_segmentations(control_points, frames, ini_frame, end_frame):

    print('number of frames: ', len(frames))
    id_frame=0

    curve_1 = BSpline.Curve()
    # Set evaluation delta (control the number of curve points)
    curve_1.delta = 0.02
    curve_1.order = 3
    curve_1.ctrlpts = control_points[id_frame]
    curve_1.knotvector = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7]

    # geometric distance between apex and base at the end of diastole (ED)
    apex = np.array(control_points[0][0])
    base = np.array(control_points[0][4])
    # print('apex', apex)
    # print('base', base)
    dist_ed = np.linalg.norm(apex - base)
    
    # radius of every circle along the curve defined in fuction of the chamber's size
    radius = np.int32(np.rint(dist_ed / 20.0))

    print('dist radius: ', dist_ed, radius)

    # mask for intensities extraction inside the circles along the curve
    ref_mask = np.zeros((448,448),np.uint8)
    ref_thresh = np.zeros((448,448),np.uint8)

    features_intensities = []

    # control points for each frame
    for id_frame, ctrl_pts in zip(np.arange(len(control_points)), control_points):

        # print('data frame ctrlpts')
        # print(id_frame)
        # print(ctrl_pts)
        # frame visualization
        img = np.copy(frames[id_frame])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img,(448,448),interpolation=cv2.INTER_CUBIC)

        # print('image size:', img.shape)
        # updating control points due to mouse interaction
        curve_1.ctrlpts = ctrl_pts
        
        values_circles = []
        # circles centered in each point along the Spline curve
        for coord_point in curve_1.evalpts:
            # intensity values inside the circles. We would need a mask of black
            mask_circle = np.copy(ref_mask)
            # painting a fulled white circle in the black mask
            cv2.circle(mask_circle, np.int32(coord_point), radius, 255, -1)
            ## images visualization
            
            img_roi = np.ma.array(img, mask=cv2.bitwise_not(mask_circle))
            # taking values inside the circle
            values_circles.append(img_roi.compressed().tolist())
            # print('shape: ', img_roi.compressed().shape)
            # print('img_roi mean:',img_roi.mean())
            # print('img_roi median:',np.ma.median(img_roi))

        # print('values_circles.shape:', values_circles.shape)
        # print('mean: ', values_circles.mean())
        if id_frame==0:
            threshold_value = np.median(values_circles)
            print('threshold value:', threshold_value)
        else:
            pass

        # thresholding, counting pixels above the threshold, and normalizing with the total number of pixels in each region
        feature_values = np.count_nonzero(values_circles > threshold_value, axis=1) / len(values_circles[0])

        # print('frame: ', id_frame)
        # print(len(feature_values), len(values_circles[0]))
        # print(feature_values)

        # feature vector for all the frames
        features_intensities.append(feature_values.tolist())


    return features_intensities


####### main function ###########
if __name__== '__main__':

    filename_controlpts = 'data/controlpts_segmentations'

    # reading list of segmented sequences
    with open(filename_controlpts, 'rb') as fp:
        # the frames count started at 1 (first frame)
        print('reading data ...')
        [file_name_list, control_points_list] = pickle.load(fp)
        print ('done.')
    print(file_name_list)

    
    ## read file name from a csv file
    print('read csv file')

    data_coord = pd.read_csv('../VolumeTracings.csv')
    print(data_coord.head())

    print('testing reading section...')
    ## read the video of ultrasound images using file_name
    dir_videos = '../Videos/'
    ini_cont_rows = 0
    for fn, cpts in zip(file_name_list[0:2], control_points_list):
        print('cont, file name: ', ini_cont_rows, fn)
        while fn == data_coord.at[ini_cont_rows,'FileName']:
            ### estimation and visualization image segmentations
            file_name, frames, ini_frame, end_frame, cont_rows = read_frames(dir_videos, data_coord, ini_cont_rows)
            # print('filename, cont_rows:', file_name, cont_rows)    
            frames_cpts = segmentation_interpolation(cpts, frames)
            feat_intens = drawing_segmentations(frames_cpts, frames, ini_frame, end_frame)

            print('features intensities')
            print(len(feat_intens))
            print(feat_intens[0])
            print(len(feat_intens[0]))

            # estimation and visualization image segmentations ###
            ini_cont_rows=cont_rows
    print('cont, file name: ', ini_cont_rows, data_coord.at[ini_cont_rows,'FileName'])
    print('testing reading section... done.')

