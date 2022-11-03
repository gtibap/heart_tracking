# particle filter for motion tracking and estimation of the End of Systole left ventricular cavity
import numpy as np
import cv2
import pickle
import pandas as pd
from geomdl import BSpline
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from numpy.random import randn

def drawing_curve(frame, control_points):

    # drawing an initial B-Spline curve in a ED frame of a the test data set
    curve = BSpline.Curve()
    # Set evaluation delta (control the number of curve points)
    curve.delta = 0.02
    curve.order = 3
    # curve.ctrlpts = parameters.tolist()
    curve.ctrlpts = control_points
    curve.knotvector = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7]

    img = np.copy(frame)
    # resizing for a better image visualization
    img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img,(448,448),interpolation=cv2.INTER_CUBIC)
    # drawing control points
    # for ctl_pt in curve.ctrlpts:
    #     cv2.circle(img, np.int32(ctl_pt), 3, (0,0,255), -1)
    # drawing the curve 
    cv2.polylines(img, np.int32([curve.evalpts]), 0, (0,255,255), 1)
    # cv2.polylines(img, np.int32([curve_es.evalpts]), 0, (0,255,0), 1)
    # drawing the main axis of the segmented region on the image
    # cv2.line(img, np.int32(curve.ctrlpts[0]), np.int32(curve.ctrlpts[4]), (255,255,0), 1)

    # cv2.imshow('ED',img)
    # cv2.waitKey(100)
    return

def rect2pol(coord):
    rho = np.sqrt(coord[0]**2 + coord[1]**2)
    phi = np.arctan2(coord[1], coord[0])
    return np.array([rho, phi])

def pol2rect(coord):
    x = coord[0] * np.cos(coord[1])
    y = coord[0] * np.sin(coord[1])
    return np.array([x, y])


def predict(particles, u, std, dt=1.):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)`"""

    N = len(particles)
    # update heading
    # print('heading:')
    # print(particles[:, 2])
    particles[:, 2] += u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi
    # print(particles[:, 2])

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (randn(N) * std[1])
    # print('dist:')
    # print(dist)
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist
    # print('particles:')
    # print(particles)

# def create_particles(ctrlpts, mean, std):
#     particles=[]


    return particles

# generating particles (sets of control points) for boundaries' estimation of the left ventricular cavity
# N number of particles
def create_particles(control_points, N):

    control_points = np.array(control_points)

    # print('len ocntro pots: ', len(control_points))
    # control points 0 and 4 define main axis in each of the two segmented images
    main_axis_ini = control_points[4] - control_points[0]
    # main_axis_end = control_points[1][4] - control_points[1][0]

    # axis normalization

    main_axis_ini = main_axis_ini / np.linalg.norm(main_axis_ini)
    # main_axis_end = main_axis_end / np.linalg.norm(main_axis_end)

    # defining a new main axis that depends of the number of frames between the segmented images

    # diff_main_axis = main_axis_end - main_axis_ini
    # diff_main_axis = rect2pol(main_axis_end) - rect2pol(main_axis_ini)

    # the origin (apex location) is also changing from frame to frame
    # defining origin for each frame (new_origin)
    # diff_origins = control_points[1][0] - control_points[0][0]


    # control points are also changing from frame to frame
    # calculating the coordinates of each control point on its respective frame of reference
    # first: subtract origin for the respective first and end frames
    pts = control_points - control_points[0]
    # pts_end = control_points[1] - control_points[1][0]
    # print(control_points[0])
    # print(pts_ini) 
    #second: calculating components of each resultant vector in the main_axis frame of reference
    # calculating secundary axis that is perpendicular to the main axis for the initial and end frames
    secu_axis_ini = np.cross(main_axis_ini, (0,0,-1))[0:2]
    # secu_axis_end = np.cross(main_axis_end, (0,0,-1))[0:2]

    xs_ini = np.dot(pts, secu_axis_ini)
    ys_ini = np.dot(pts, main_axis_ini)
    
    xs_ini = np.reshape(xs_ini, (-1,1))
    ys_ini = np.reshape(ys_ini, (-1,1))

    pts_ctl = np.concatenate((xs_ini,ys_ini),axis=1)

    # print('xs', xs_ini)
    # print('ys', ys_ini)
    # print('[xs,ys]', pts_ctl, pts_ctl.shape)

    # particles = np.empty((N, len(control_points), 2))
    particles = np.repeat([pts_ctl], N, axis=0) 
    # print(len(particles), particles.shape)
    # print(particles)
    # particles[:, 1] = mean[1] + (randn(N) * std[1])
    # particles[:, 2] = mean[2] + (randn(N) * std[2])
    # particles[:, 2] %= 2 * np.pi


    return particles

    # xs_end = np.dot(pts_end, secu_axis_end)
    # ys_end = np.dot(pts_end, main_axis_end)

    # third: calculating deltas between two consecutive frames
    # diff_xs = xs_end - xs_ini
    # diff_ys = ys_end - ys_ini
    
    # # calcualting ratio of increments amount frames using a cosine function
    # alphas = np.linspace(0, 3.14159265, len(frames))
    # ratios = (1-np.cos(alphas)) / 2.0

    # # frames_control_pts_1 = []
    # main_axis_polar = rect2pol(main_axis)
    # # the main axis is changing from frame to frame
    # for frame_number in np.arange(len(frames)):

    #     # new_main_axis_1 = main_axis_ini_polar + diff_main_axis*ratios[frame_number]
    #     new_main_axis = main_axis_polar + delta_ang
    #     new_main_axis = pol2rect(new_main_axis)
    #     # print('y_new:', new_main_axis)

    #     # main axis is y axis, new_secu_axis is x axis, and (0,0,-1) is z axis
    #     # applying cross product and taking only two components of the result
    #     new_secu_axis_1 = np.cross(new_main_axis_1, (0,0,-1))[0:2]
    #     # print('x_new:', new_secu_axis)

    #     # the origin (apex location) is also changing from frame to frame
    #     # defining origin for each frame (new_origin)
    #     new_origin_1 = control_points[0][0] + diff_origins*ratios[frame_number]

    #     # fourth: adding deltas according to the frame number
    #     new_xs_1 = xs_ini + diff_xs*ratios[frame_number]
    #     new_ys_1 = ys_ini + diff_ys*ratios[frame_number]


    #     # reshaping to apply matrix multiplication
    #     # assembling all to calculate coordinates in the original frame of reference
    #     new_main_axis_1 = np.reshape(new_main_axis_1, (1,2))
    #     new_secu_axis_1 = np.reshape(new_secu_axis_1, (1,2))

    #     new_xs_1 = np.reshape(new_xs_1, (-1,1))
    #     new_ys_1 = np.reshape(new_ys_1, (-1,1))

    #     new_pts_1 = new_xs_1*new_secu_axis_1 + new_ys_1*new_main_axis_1 + new_origin_1

    #     # frames_control_pts_0.append(new_pts_0.tolist())
    #     frames_control_pts_1.append(new_pts_1.tolist())


    # return frames_control_pts_1


#### function read video ultrasound ####
def read_frames(dir_video, seg_data):

    # for ind, row in seg_data.iterrows():
    #     print(ind, row['FileName'], row['Frame'])

    selected_frames = []
    
    # print('indexes:', seg_data.index)
    cont_rows = seg_data.index[0]
    end_rows =  seg_data.index[-1]
    frame_number = seg_data.at[cont_rows, 'Frame']
    new_frame_number = seg_data.at[cont_rows, 'Frame']
    
    # initial frame id
    id_frame_ini = frame_number

    cap = cv2.VideoCapture(dir_video)

    segmented_frames = 0
    save_frames = False
    cont_frames=0
    end_seg_data=False
    scale_factor = 4 

    while cap.isOpened() and segmented_frames < 2:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if cont_frames == frame_number:

            img = np.copy(frame)
            # resizing for a better image visualization
            img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
            img = cv2.resize(img,(448,448),interpolation=cv2.INTER_CUBIC)

            # drawing image segmentation--lines in the left ventricle--on a selected frame (end of diastole or end of systole)
            while frame_number == new_frame_number and end_seg_data==False:

                x1 = round(scale_factor * seg_data.at[cont_rows,'X1'])
                y1 = round(scale_factor * seg_data.at[cont_rows, 'Y1'])

                x2 = round(scale_factor * seg_data.at[cont_rows,'X2'])
                y2 = round(scale_factor * seg_data.at[cont_rows, 'Y2'])

                # drawing image segmentation, line by line
                cv2.line(img,(x1,y1),(x2,y2),(255,255,0),1)

                cont_rows+=1
                if cont_rows <= end_rows:
                    new_frame_number = seg_data.at[cont_rows,'Frame']
                else:
                    end_seg_data=True

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

    return selected_frames, frame_ini, frame_end


def display_frames(frames):
    id_frame=0
    selected_image = 0
    flag_continue = True
    while flag_continue:

        # frame visualization
        img = np.copy(frames[id_frame])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img,(448,448),interpolation=cv2.INTER_CUBIC)


        ## images visualization
        cv2.imshow('frame',img)
        k = cv2.waitKey(100) & 0xFF

        # move forward, next frame
        if k == ord('f') and id_frame<(len(frames)-1):
            # mode = not mode
            id_frame=id_frame+1
            # print('id_frame: ', id_frame)
        # move backward, previous frame
        elif k == ord('b') and id_frame>0:
            id_frame=id_frame-1
            # print('id_frame: ', id_frame)           

        # close window (scape key)
        elif k == 27:
            flag_continue = False

        else:
            pass
    
    return 0



    

####### main function ###########
if __name__== '__main__':

    filename_controlpts = 'data/controlpts_segmentations'
    filename_pca = 'data/pca_data'
    dir_video = '../Videos/'
    filename_segmentations = '../VolumeTracings.csv'
    filename_features_ctrlpts = 'data/features_ctlpts.csv'

    # reading list of intensity features
    with open(filename_pca, 'rb') as fp:
        # the frames count started at 1 (first frame)
        print('reading pca data ...')
        [X_train, X_test, names_train, names_test, pca_list, mean_list, std_list] = pickle.load(fp)
        print ('done.')


    # reading features control points
    with open(filename_controlpts, 'rb') as fp:
        # the frames count started at 1 (first frame)
        print('reading control points B-Splines ...')
        [file_name_list, control_points_list] = pickle.load(fp)
        print ('done.')
    df_controlpts = pd.DataFrame(control_points_list)
    df_controlpts.rename(columns={0:'ED',1:'ES'}, inplace=True)
    df_controlpts['FileName']=file_name_list
    # print(df_controlpts.head())

    # print('filenames: ', len(file_name_list), file_name_list)
    # print('ctrlpts: ', len(control_points_list), len(control_points_list[0]), len(control_points_list[0][0]),control_points_list[0][0])
    # print('ctrlpts: ', len(control_points_list), len(control_points_list[1]), len(control_points_list[1][0]),control_points_list[1][0])


    # save features control points
    print('reading features from control points...')
    df_feat_ctrlpts = pd.read_csv(filename_features_ctrlpts)
    print('done.')
    # print(df_feat_ctrlpts.head())

    # training and test data sets
    # reading each training video
    train_cp = pd.DataFrame()
    for filename in names_train:
        selected = df_feat_ctrlpts[df_feat_ctrlpts['filename']==filename]
        train_cp = pd.concat([train_cp, selected])
    # print(train_cp.head())

    # reading each test video
    test_cp = pd.DataFrame()
    for filename in names_test:
        selected = df_feat_ctrlpts[df_feat_ctrlpts['filename']==filename]
        test_cp = pd.concat([test_cp, selected])
    # print(test_cp.head())

    # train_cp[]
    # df_subset = train_cp[['ang','x1','x2','x3','x5','x6','x7','y1','y2','y3','y4','y5','y6','y7','c0','c1']]
    mean_cp = train_cp[['ang','x1','x2','x3','x5','x6','x7','y1','y2','y3','y4','y5','y6','y7','c0','c1']].mean()
    std_cp = train_cp[['ang','x1','x2','x3','x5','x6','x7','y1','y2','y3','y4','y5','y6','y7','c0','c1']].std()
    # print(mean_cp)
    # print(std_cp)


    # binwidth=0.1
    # min=-2.0
    # max=2.0
    # train_cp.hist(column=['ang','x1','x2','x3','x5','x6','x7','y1','y2','y3','y4','y5','y6','y7','c0','c1'], bins=np.arange(min, max + binwidth, binwidth))
    # # train_cp.hist()
    # plt.show()
   

    # # applying PCA to the training data
    # pca_cp=PCA(n_components=13)
    # train_sel = train_cp[['x1','x2','x3','x5','x6','x7','y1','y2','y3','y4','y5','y6','y7','c0','c1']].values.tolist()
    # # print(len(train_sel), len(train_sel[0]), train_sel[0])

    # # pca_cp.fit(train_sel)
    # pca_train_sel = pca_cp.fit_transform(train_sel)
    # rec_train_sel = pca_cp.inverse_transform(pca_train_sel)

    # print('org:', train_sel[0])
    # print('rec:', rec_train_sel[0])

    # acc = np.cumsum(pca_cp.explained_variance_ratio_)
    # # print(pca_cp.singular_values_)
    # print(pca_cp.explained_variance_ratio_)
    # print(acc)
    # # print(feat_single.shape)
    # # df = pd.DataFrame(pca_train_sel)
    # # binwidth=0.05
    # # min=-3.0
    # # max=3.0
    # # df.hist(bins=np.arange(min, max + binwidth, binwidth))
    # # # df.hist()
    # # plt.show()
    

    # print(len(X_train), len(X_test), len(pca_list), len(mean_list), len(std_list))
    # print(names_test)

    # read test data set
    ## read file name from a csv file
    print('reading image segmentation of reference from Echonet...')
    df_seg = pd.read_csv(filename_segmentations)
    # data_coord = pd.read_csv(filename_segmentations)
    print('done.')
    # print(data_coord.head())

    # reading each test video
    for id_test, filename_test in enumerate(names_test[0:1]):

        # opening test data: ED and ES frames with their manual segmentations; frames between ED and ES
        # manual segmentation
        df_test = df_seg[df_seg['FileName']==filename_test]
        # print(seg_data)

        
        # segmented ED and ES images
        selected_frames, frame_ini, frame_end = read_frames(dir_video + filename_test, df_test)

        # img_ed_es = np.concatenate((frame_ini,frame_end),axis=1)
        # cv2.imshow('ref', img_ed_es)
        # display_frames(selected_frames)

        # opening control points for the test video
        # features_intensities = X_test[0]
        # print(len(features_intensities), len(features_intensities[0]))
        # print(ctrl_points)

        # control points of the first test video
        test_df_controlpts = df_controlpts[df_controlpts['FileName']==filename_test]
        # print(test_df_controlpts)
        ctrlpts_test_ed = test_df_controlpts['ED'].values.tolist()[0]
        ctrlpts_test_es = test_df_controlpts['ES'].values.tolist()[0]

        control_points = np.array(ctrlpts_test_ed)
        print(control_points)
        # control points 0 and 4 define main axis in ED
        main_axis_ini = control_points[4] - control_points[0]
        # normalization factor w
        factor = 5.0
        w = np.linalg.norm(main_axis_ini) / factor
        print('w:',w)
        
        
        # # drawing an initial B-Spline curve in a ED frame of a the test data set
        # curve_es = BSpline.Curve()
        # # Set evaluation delta (control the number of curve points)
        # curve_es.delta = 0.02
        # curve_es.order = 3
        # # curve.ctrlpts = parameters.tolist()
        # curve_es.ctrlpts = ctrlpts_test_es
        # curve_es.knotvector = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7]

        # particles initialization
        particles = create_particles(ctrlpts_test_ed, 2)
        pf_control_points = []
        # frame visualization
        # img = np.copy(frame_ini)
        for frame in selected_frames:
        
            # calcualting ratio of increments amount frames using a cosine function
            # alphas = np.linspace(0, 3.14159265, len(frames))
            # ratios = (1-np.cos(alphas)) / 2.0

            # ctrlpts_sets = particles_generation(ctrlpts_test_es)
            particles = predict(particles, mean_cp, std_cp)
            # curve.ctrlpts = update()

        for frame, control_points in zip(selected_frames, pf_control_points):
            drawing_curve(frame, control_points)

        cv2.waitKey(0)

    cv2.destroyAllWindows()



    # print('indexes:', seg_data.index)
    # ind0 = seg_data.index[0]
    # print('index 0:', ind0 )

    # indL = seg_data.index[-1]
    # print('index L:', indL )
    # print(seg_data.at[ind0, 'Frame'])
    # file_name = seg_data.at[0,'FileName']
    # for ind in seg_data.index:
    #     print('index:', ind, seg_data.at[ind,'FileName'], seg_data.at[ind,'Frame'])

    # for ind, row in seg_data.iterrows():
    #     print(ind, row['FileName'], row['Frame'])






