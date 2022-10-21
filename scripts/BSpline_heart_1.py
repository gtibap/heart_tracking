#from symbol import parameters
from symbol import parameters
import numpy as np 
import SimpleITK as sitk
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

# mouse callback function
def mouse_interact(event,x,y,flags,param):
    global parameters, drawing, id_point, id_frame
    
    # print('id_frame: ', id_frame)
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # here we choose the closest point to the mouse pointer
        delta_dist = np.subtract(parameters, (x,y))
        mag_delta = LA.norm(delta_dist, axis=1)
        # print('mag_delta: ', mag_delta)
        id_point = np.argmin(mag_delta)

    elif event == cv2.EVENT_MOUSEMOVE and drawing == True:
        parameters[id_point] = (x,y)
        # first and last control points share same coordinates
        parameters[-1] = parameters[0]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    
    else:
        pass
        
    return

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
                print('segmented image: ',file_name, frame_number)
                print('frame number: ', cont_frames)
            else:
                # saving second (last) segmented frame
                frame_end = np.copy(img)
                selected_frames.append(frame)
                save_frames = False
                print('segmented image: ',file_name, frame_number)
                print('frame number: ', cont_frames)

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

    # print('main axes: ')
    # print(main_axis_ini)
    # print(main_axis_end)

    # defining a new main axis that depends of the number of frames between the segmented images
    # first coordinates transformation rect to polar 
    # rect2pol(main_axis_ini)
    # rect2pol(main_axis_end)
    
    # diff_main_axis = main_axis_end - main_axis_ini
    diff_main_axis = rect2pol(main_axis_end) - rect2pol(main_axis_ini)
    
    # delta_main_axis = diff_main_axis / (len(frames)-1)
    print("diff_main_axis")
    print(diff_main_axis)
    # print(delta_main_axis)

    # the origin (apex location) is also changing from frame to frame
    # defining origin for each frame (new_origin)
    diff_origins = control_points[1][0] - control_points[0][0]
    # delta_origins = diff_origins / (len(frames)-1)

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

    # delta_xs = diff_xs / (len(frames)-1)
    # delta_ys = diff_ys / (len(frames)-1)

    # print('control points first frame')
    # print(control_points[0])
    
    # calcualting ratio of increments amount frames using a cosine function
    alphas = np.linspace(0, 3.14159265, len(frames))
    ratios = (1-np.cos(alphas)) / 2.0

    # frames_control_pts_0 = []
    frames_control_pts_1 = []
    # the main axis is changing from frame to frame
    print("new main axis 1")
    for frame_number in np.arange(len(frames)):

        # new_main_axis_0 = main_axis_ini + delta_main_axis*frame_number
        # new_main_axis_1 = main_axis_ini + diff_main_axis*ratios[frame_number]
        new_main_axis_1 = rect2pol(main_axis_ini) + diff_main_axis*ratios[frame_number]
        # print(new_main_axis_1)
        new_main_axis_1 = pol2rect(new_main_axis_1)
        # print('y_new:', np.linalg.norm(new_main_axis_1))

        # new_main_axis_1 = new_main_axis_1 / np.linalg.norm(new_main_axis_1)
        # print('y_new:', np.linalg.norm(new_main_axis_1))

        # main axis is y axis, new_secu_axis is x axis, and (0,0,-1) is z axis
        # applying cross product and taking only two components of the result
        # new_secu_axis_0 = np.cross(new_main_axis_0, (0,0,-1))[0:2]
        new_secu_axis_1 = np.cross(new_main_axis_1, (0,0,-1))[0:2]
        # print('x_new:', new_secu_axis)

        # the origin (apex location) is also changing from frame to frame
        # defining origin for each frame (new_origin)
        # delta_origins = (control_points[1][0] - control_points[0][0]) / (len(frames)-1)
        # new_origin_0 = control_points[0][0] + delta_origins*frame_number
        new_origin_1 = control_points[0][0] + diff_origins*ratios[frame_number]

        # print(control_points[0][0])
        # print(control_points[1][0])
        # print(diff_origins)
        # print(delta_origins)
        # print(new_origin)

        # fourth: adding deltas according to the frame number
        # new_xs_0 = xs_ini + delta_xs*frame_number
        # new_ys_0 = ys_ini + delta_ys*frame_number
        new_xs_1 = xs_ini + diff_xs*ratios[frame_number]
        new_ys_1 = ys_ini + diff_ys*ratios[frame_number]


        # reshaping to apply matrix multiplication
        # assembling all to calculate coordinates in the original frame of reference
        # new_main_axis_0 = np.reshape(new_main_axis_0, (1,2))
        # new_secu_axis_0 = np.reshape(new_secu_axis_0, (1,2))
        new_main_axis_1 = np.reshape(new_main_axis_1, (1,2))
        new_secu_axis_1 = np.reshape(new_secu_axis_1, (1,2))

        # new_xs_0 = np.reshape(new_xs_0, (-1,1))
        # new_ys_0 = np.reshape(new_ys_0, (-1,1))
        new_xs_1 = np.reshape(new_xs_1, (-1,1))
        new_ys_1 = np.reshape(new_ys_1, (-1,1))

        # new_pts_0 = new_xs_0*new_secu_axis_0 + new_ys_0*new_main_axis_0 + new_origin_0
        new_pts_1 = new_xs_1*new_secu_axis_1 + new_ys_1*new_main_axis_1 + new_origin_1

        # print(control_points[0])
        # print(frame_number)
        # print(new_pts)
        # frames_control_pts_0.append(new_pts_0.tolist())
        frames_control_pts_1.append(new_pts_1.tolist())

    # print('control points last frame')
    # print(control_points[1])
    # print(new_xs)
    # print(new_ys)

    #

    # transoforming the pts to the original frame of reference
    # print(x_comp_ini)
    # print(y_comp)
    # print(secu_axis_ini)
    # print(main_axis_ini)

    # main_axis_ini = np.reshape(main_axis_ini,(1,2))
    # x_comp_ini = np.reshape(x_comp_ini, (-1,1))
    # y_comp = np.reshape(y_comp, (-1,1))

    # # print(main_axis_ini.shape)
    # # print(x_comp_ini.shape)
    # # print(y_comp.shape)

    # pts_inia = x_comp_ini*secu_axis_ini + y_comp*main_axis_ini

    # print(pts_ini)
    # print(pts_inia)


    return frames_control_pts_1


# drawing image segmentations on all selected frames of each video
def drawing_segmentations(control_points, frames, ini_frame, end_frame):

    # drawing initial and end segmented frames
    two_images = np.concatenate((ini_frame,end_frame),axis=1)
    ## images visualization
    cv2.imshow('images_ref',two_images)
    # k = cv2.waitKey(1) & 0xFF

    id_frame=0
    # curve_0 = BSpline.Curve()
    # # Set evaluation delta (control the number of curve points)
    # curve_0.delta = 0.02
    # curve_0.order = 3
    # curve_0.ctrlpts = control_points[id_frame].tolist()
    # curve_0.knotvector = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7]

    curve_1 = BSpline.Curve()
    # Set evaluation delta (control the number of curve points)
    curve_1.delta = 0.02
    curve_1.order = 3
    curve_1.ctrlpts = control_points[id_frame]
    curve_1.knotvector = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7]

    # geometric distance between apex and base
    apex = np.array(control_points[0][0])
    base = np.array(control_points[0][4])
    print('apex', apex)
    print('base', base)

    # radius of every circle along the curve defined in fuction of the chamber's size
    dist_ed = np.linalg.norm(apex - base)
    radius = np.int32(dist_ed / 20.0)

    print('dist radius: ', dist_ed, radius)

    selected_image = 0
    flag_continue = True
    while flag_continue:

        # frame visualization
        img = np.copy(frames[id_frame])
        img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img,(448,448),interpolation=cv2.INTER_CUBIC)

        # if selected_image == 0:
        #     img = np.copy(ini_frame)
        #     parameters = control_points[0].tolist()
        # else:
        #     img = np.copy(end_frame)
        #     parameters = control_points[1].tolist()

        # updating control points due to mouse interaction
        # curve_0.ctrlpts = frames_control_pts_0[id_frame]
        curve_1.ctrlpts = control_points[id_frame]

        # print ('number of points: ', len(curve_1.evalpts))
        # cv2.polylines(img, np.int32([curve_1.ctrlpts]), 0, (0,255,0), 1)
        # cv2.polylines(img, np.int32([curve_0.evalpts]), 0, (0,0,255), 1)
        # cv2.polylines(img, np.int32([curve_1.evalpts]), 0, (0,255,255), 2)

        for coord_point in curve_1.evalpts:
            cv2.circle(img, np.int32(coord_point), radius, (0,255,255), 1)

        # drawing the main axis of the segmented region on the image
        cv2.line(img, np.int32(curve_1.ctrlpts[0]), np.int32(curve_1.ctrlpts[4]), (255,0,0), 1)

        ## images visualization
        cv2.imshow('image',img)
        k = cv2.waitKey(100) & 0xFF

        # move forward, next frame
        if k == ord('f') and id_frame<(len(frames)-1):
            # mode = not mode
            id_frame=id_frame+1
            print('id_frame: ', id_frame)
        # move backward, previous frame
        elif k == ord('b') and id_frame>0:
            id_frame=id_frame-1
            print('id_frame: ', id_frame)           

        # close window (scape key)
        elif k == 27:
            flag_continue = False

        else:
            pass
    
    return 0


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

    # file_name_list=[]
    # control_points_list=[]
    # with open(filename_controlpts, 'wb') as fp:
    #     # the frames count started at 1 (first frame)
    #     print('saving control points ...')
    #     pickle.dump([file_name_list, control_points_list], fp)
    #     print ('done.')    

    print('BSpline for heart segmentation')
    
    ## read file name from a csv file
    print('read csv file')

    data_coord = pd.read_csv('../VolumeTracings.csv')
    print(data_coord.head())

    # file_name_list=[]
    # control_points_list=[]
    # reading the first file name (first video)
    # file_name = data_coord.at[0,'FileName']
    # print(file_name)

    print('testing reading section...')
    ## read the video of ultrasound images using file_name
    dir_videos = '../Videos/'
    ini_cont_rows = 0
    for fn, cpts in zip(file_name_list, control_points_list):
        print('cont, file name: ', ini_cont_rows, fn)
        while fn == data_coord.at[ini_cont_rows,'FileName']:
            ### estimation and visualization image segmentations
            file_name, frames, ini_frame, end_frame, cont_rows = read_frames(dir_videos, data_coord, ini_cont_rows)
            # print('filename, cont_rows:', file_name, cont_rows)    
            frames_cpts = segmentation_interpolation(cpts, frames)
            drawing_segmentations(frames_cpts, frames, ini_frame, end_frame)
            # estimation and visualization image segmentations ###
            ini_cont_rows=cont_rows
    print('cont, file name: ', ini_cont_rows, data_coord.at[ini_cont_rows,'FileName'])
    print('testing reading section... done.')

    next_video = True 
    while next_video == True:
    
        file_name, frames, ini_frame, end_frame, cont_rows = read_frames(dir_videos, data_coord, ini_cont_rows)
        print('cont_rows:', cont_rows)

        print('number of selected frames: ', len(frames))

        # ref_frames = np.concatenate((ini_frame,end_frame),axis=1)
        # cv2.imshow('segmented images',ref_frames)
        # k = cv2.waitKey(1)

        # activate interactive mouse actions on window
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',mouse_interact)
        drawing = False

        ## curve B-spline definition
        p1, p2, p3, p4, p5, p6, p7, p8, p9 = [224, 112], [112, 168], [168, 224], [112, 280], [224, 336], [336, 280], [280, 224], [336, 168], [224, 112]
        parameters = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9])

        # print('parameters:')
        # print(parameters)

        id_frame=0
        curve = BSpline.Curve()
        # Set evaluation delta (control the number of curve points)
        curve.delta = 0.02
        curve.order = 3
        curve.ctrlpts = parameters.tolist()
        curve.knotvector = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7]

        control_points = []
        next_image = 0
        flag_continue = True

        while flag_continue:

            if next_image == 0:
                # first segmented image by an expert
                img = np.copy(ini_frame)
            else:
                # second and last segmented image by an expert
                img = np.copy(end_frame)
            
            # updating control points due to mouse interaction
            curve.ctrlpts = parameters.tolist() 
            # print ('number of points: ', len(curve.evalpts))
            for ctl_pt in curve.ctrlpts:
                cv2.circle(img, np.int32(ctl_pt), 3, (0,0,255), -1)
            
            cv2.polylines(img, np.int32([curve.evalpts]), 0, (0,255,255), 1)

            # drawing the main axis of the segmented region on the image
            cv2.line(img, np.int32(curve.ctrlpts[0]), np.int32(curve.ctrlpts[4]), (255,255,0), 1)

            ## images visualization
            cv2.imshow('image',img)
            k = cv2.waitKey(100) & 0xFF

            if k == ord('n'):
                control_points.append(parameters.tolist())
                print('appended control points for image : ', next_image)
                next_image+=1

            # save parameters: control points B-Spline for every frame
            elif k == ord('x'):
                file_name_list.append(file_name)
                control_points_list.append(control_points)
                with open(filename_controlpts, 'wb') as fp:
                    # the frames count started at 1 (first frame)
                    print('saving control points ...')
                    pickle.dump([file_name_list, control_points_list], fp)
                    print ('done.')

            # close window (scape key)
            elif k == 27:
                flag_continue = False

            else:
                pass

            #print ('center, scale, rotation: ', center, ', ', long_ab, ', ', rotation)
        
        print('next video? (Y/n)')
        k1 = cv2.waitKey(0) & 0xFF
        if k1 == ord('n'):
            next_video=False
        else:
            ini_cont_rows = cont_rows

    cv2.destroyAllWindows()




