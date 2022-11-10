# particle filter for motion tracking and estimation of the End of Systole left ventricular cavity
import sys
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
import scipy.stats
from filterpy.monte_carlo import systematic_resample


def neff(weights):
    return 1. / np.sum(np.square(weights))


def resample_from_index(particles, weights, indexes):
    # print('indexes:', len(indexes), indexes)
    particles[:] = particles[indexes]
    # print('len(particles): ', len(particles))
    weights.resize(len(particles))
    weights.fill (1.0 / len(weights))
    return weights


def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""
    # pos = particles[:, 0:2]
    mean = np.average(particles, weights=weights, axis=0)
    var  = np.average((particles - mean)**2, weights=weights, axis=0)
    return mean, var
    # pos = particles[:, 0:2]
    # mean = np.average(pos, weights=weights, axis=0)
    # var  = np.average((pos - mean)**2, weights=weights, axis=0)
    # return mean, var


# drawing image segmentations on all selected frames of each video
def function_intensities(control_points, frame, id_frame, radius, threshold_value):

    # print('number of frames: ', len(frames))
    # id_frame=0

    curve_1 = BSpline.Curve()
    # Set evaluation delta (control the number of curve points)
    curve_1.delta = 0.02
    curve_1.order = 3
    curve_1.ctrlpts = control_points
    curve_1.knotvector = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7]

    # # geometric distance between apex and base at the end of diastole (ED) (first frame)
    # apex = np.array(control_points[0])
    # base = np.array(control_points[4])
    # # print('apex', apex)
    # # print('base', base)
    # dist_ed = np.linalg.norm(apex - base)

    # radius of every circle along the curve defined in fuction of the chamber's size
    # radius = np.int32(np.rint(dist_ed / 20.0))

    # print('dist radius: ', dist_ed, radius)

    # mask for intensities extraction inside the circles along the curve
    ref_mask = np.zeros((448,448),np.uint8)
    # ref_thresh = np.zeros((448,448),np.uint8)
    # image size extension in case the center of the circles are too close to the borders
    ref_mask = cv2.copyMakeBorder(ref_mask, radius,radius,radius,radius, cv2.BORDER_REPLICATE)

    # features_intensities = []

    # control points for each frame
    # for id_frame, ctrl_pts in zip(np.arange(len(control_points)), control_points):

    # print('data frame ctrlpts')
    # print(id_frame)
    # print(ctrl_pts)
    # frame visualization
    img = np.copy(frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img,(448,448),interpolation=cv2.INTER_CUBIC)
    # image size extension in case the circle too close to the image border
    img = cv2.copyMakeBorder(img, radius,radius,radius,radius,cv2.BORDER_REFLECT_101)

    # print('image size:', img.shape)
    # updating control points due to mouse interaction
    # curve_1.ctrlpts = control_points

    values_circles = []
    # circles centered in each point along the Spline curve
    for coord_point in curve_1.evalpts:
        # intensity values inside the circles. We would need a mask of black
        mask_circle = np.copy(ref_mask)
        # painting a fulled white circle in the black mask
        cv2.circle(mask_circle, np.int32(coord_point+radius), radius, 255, -1)
        ## images visualization

        # new_mask_circle = cv2.copyMakeBorder(mask_circle, radius,radius,radius,radius,cv2.BORDER_REFLECT_101)
        # cv2.circle(mask_circle, np.int32(coord_point), np.int32(radius/2), 125, -1)
        # if id_frame==0:
        #     img_2 = np.copy(img)
        #     cv2.circle(img_2, np.int32(coord_point+radius), radius, 255, 1)
        #     cv2.imshow('mask0', mask_circle)
        #     cv2.imshow('img 2', img_2)
        #     cv2.waitKey(1000)
        # else:
        #     pass


        img_roi = np.ma.array(img, mask=cv2.bitwise_not(mask_circle))
        # taking values inside the circle
        values_circles.append(img_roi.compressed().tolist())
        # print('shape: ', img_roi.compressed().shape)
        # print('img_roi mean:',img_roi.mean())
        # print('img_roi median:',np.ma.median(img_roi))

    # print('values_circles.shape:', values_circles.shape)
    # print('mean: ', values_circles.mean())
    if id_frame==0:
        # print('values_circles: ', len(values_circles), len(values_circles[0]))
        # print(values_circles)
        # avoiding zeros and taking only the region septum
        # mask_values_circles = np.ma.masked_equal(values_circles[0:25], 0)
        # print('masked median:', np.ma.median(mask_values_circles))

        # print('selected values_circles: ', len(selected_values_circles), len(selected_values_circles[0]))
        # threshold_value = np.median(values_circles)
        threshold_value = np.median(values_circles)
        # threshold_value =  np.ma.median(mask_values_circles)
        print('threshold value:', threshold_value)
        # visualization
        # img_t = np.copy(frames[id_frame])
        # img_t = cv2.resize(img_t,(224,224),interpolation=cv2.INTER_CUBIC)
        # img_t = cv2.resize(img_t,(448,448),interpolation=cv2.INTER_CUBIC)
        # for center_ci in curve_1.evalpts:
        #     cv2.circle(img_t, np.int32(center_ci), radius, (0,255,0), 1)
        # cv2.imshow('image',img_t)
        # cv2.waitKey(0)
    else:
        pass

    # thresholding, counting pixels above the threshold, and normalizing with the total number of pixels in each region
    feature_values = np.count_nonzero(values_circles > threshold_value, axis=1) / len(values_circles[0])

    # print('frame: ', id_frame)
    # print(len(feature_values), len(values_circles[0]))
    # print(feature_values)

    # feature vector for all the frames
    # features_intensities.append(feature_values.tolist())


    # return features_intensities
    return feature_values.tolist(), threshold_value



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

    return img

def rect2pol(coord):
    rho = np.sqrt(coord[0]**2 + coord[1]**2)
    phi = np.arctan2(coord[1], coord[0])
    return np.array([rho, phi])

def pol2rect(coord):
    x = coord[0] * np.cos(coord[1])
    y = coord[0] * np.sin(coord[1])
    return np.array([x, y])

def drawing_particles(particles):

    # img = np.copy(frame)
    # # resizing for a better image visualization
    # # aditionally, all data have been recording with an image size of (448,448)
    # img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
    # img = cv2.resize(img,(448,448),interpolation=cv2.INTER_CUBIC)

    ctrlpts_particles=[]

    for particle in particles:
        # print(particle)
        ang = particle[0]
        xs = particle[1:10]
        ys = particle[10:19]
        c0 = particle[19]
        c1 = particle[20]
        # print('particle')
        # print(ang, xs, ys, c0, c1)

        main_axis = pol2rect([1.0, ang])
        secu_axis = np.cross(main_axis, (0,0,-1))[0:2]

                # reshaping to apply matrix multiplication
        # assembling all to calculate coordinates in the original frame of reference
        main_axis = np.reshape(main_axis, (1,2))
        secu_axis = np.reshape(secu_axis, (1,2))

        xs = np.reshape(xs, (-1,1))
        ys = np.reshape(ys, (-1,1))

        ctrlpts = xs*secu_axis + ys*main_axis + [c0,c1]

        ctrlpts_particles.append(ctrlpts.tolist())

        # print('control points')
        # print(ctrlpts)

        # cv2.imshow('frame', img)
        # cv2.waitKey(500)

    return ctrlpts_particles


# def update(particles, weights, mean, std, frame):
def update(features_particles, df_pca_intensities, components, weights):
    # print('weights:')
    # ctrlpts_particles = drawing_particles(particles)

    # for ctrlpts in ctrlpts_particles:
    #     feat_intensities, threshold_value = function_intensities(ctrlpts, frame, id_frame, radius, threshold_value)
    #     img = drawing_curve(frame, ctrlpts)
    #     cv2.imshow('frame',img)
    #     cv2.waitKey(100)

    # print(df_pca_intensities)
    pca = df_pca_intensities['pca']
    mean_pca = df_pca_intensities['mean']
    std_pca = df_pca_intensities['std']

    # print(mean_pca)
    # print(std_pca)

    # print(len(feat_particles), len(feat_particles[0]))
    features_pca = pca.transform(features_particles)
    # print(len(features_pca), len(features_pca[0]))
    # print('features pca:')
    # print(features_pca)


    scores_particles = scipy.stats.norm(mean_pca, std_pca).pdf(features_pca)
    # print('scores:')
    # print(scores_particles)

    # print('weights: ', weights)
    # components=15
    print('components: ', components)
    for id_part, scores_part in enumerate(scores_particles):
        weights[id_part]*= np.prod(scores_part[:components])
    # print('weights: ', weights)

    # for features_one_particle in features_pca:
    #     for comp, mean_comp, std_comp in zip(features_one_particle, mean_pca, std_pca):
    #         weights *= scipy.stats.norm(mean_comp, std_comp).pdf(comp)
    # #     print()

    # for i, landmark in enumerate(landmarks):
    #     distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
    #     weights *= scipy.stats.norm(distance, R).pdf(z[i])
    #     # print(distance)
    #     # print(weights)

    # print('weightsUpdate:')
    # print(weights)
    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize
    # print(weights)

    return weights


def predict(particles, u, std):

    # print('predict')
    # # print('ang:', particles[:,0])
    # print(len(particles), len(particles[0]))
    # N = len(particles)

    # ang
    particles[:,0] += u['ang'] + (randn(N) * std['ang'])
    # # x0
    # particles[:,1] += u[1] + (randn(N) * std[1])
    # x1
    particles[:,2] += u['x1'] + (randn(N) * std['x1'])
    # x2
    particles[:,3] += u['x2'] + (randn(N) * std['x2'])
    # x3
    particles[:,4] += u['x3'] + (randn(N) * std['x3'])
    # # x4
    # particles[:,5] += u[5] + (randn(N) * std[5])
    # x5
    particles[:,6] += u['x5'] + (randn(N) * std['x5'])
    # x6
    particles[:,7] += u['x6'] + (randn(N) * std['x6'])
    # x7
    particles[:,8] += u['x7'] + (randn(N) * std['x7'])
    # # x8
    # particles[:,9] += u[9] + (randn(N) * std[9])
    # # y0
    # particles[:,10] += u[10] + (randn(N) * std[10])
    # y1
    particles[:,11] += u['y1'] + (randn(N) * std['y1'])
    # y2
    particles[:,12] += u['y2'] + (randn(N) * std['y2'])
    # y3
    particles[:,13] += u['y3'] + (randn(N) * std['y3'])
    # y4
    particles[:,14] += u['y4'] + (randn(N) * std['y4'])
    # y5
    particles[:,15] += u['y5'] + (randn(N) * std['y5'])
    # y6
    particles[:,16] += u['y6'] + (randn(N) * std['y6'])
    # y7
    particles[:,17] += u['y7'] + (randn(N) * std['y7'])
    # # y8
    # particles[:,18] += u[18] + (randn(N) * std[18])
    # c0
    particles[:,19] += u['c0'] + (randn(N) * std['c0'])
    # c1
    particles[:,20] += u['c1'] + (randn(N) * std['c1'])

    return particles

    # for  id in np.arange(len(particles[0])):
    #     particles[:,id] += u[id] + (randn(N) * std[id])

    # return particles

    # # ang
    # particles[:,0] += u[0] + (randn(N) * std[0])
    # # x0
    # particles[:,1] += u[1] + (randn(N) * std[1])
    # # x1
    # particles[:,2] += u[2] + (randn(N) * std[2])
    # # x2
    # particles[:,3] += u[3] + (randn(N) * std[3])
    # # x3
    # particles[:,4] += u[4] + (randn(N) * std[4])
    # # x4
    # particles[:,5] += u[5] + (randn(N) * std[5])
    # # x5
    # particles[:,6] += u[6] + (randn(N) * std[6])
    # # x6
    # particles[:,7] += u[7] + (randn(N) * std[7])
    # # x7
    # particles[:,8] += u[8] + (randn(N) * std[8])
    # # x8
    # particles[:,9] += u[9] + (randn(N) * std[9])
    # # y0
    # particles[:,10] += u[10] + (randn(N) * std[10])
    # # y1
    # particles[:,11] += u[11] + (randn(N) * std[11])
    # # y2
    # particles[:,12] += u[12] + (randn(N) * std[12])
    # # y3
    # particles[:,13] += u[13] + (randn(N) * std[13])
    # # y4
    # particles[:,14] += u[14] + (randn(N) * std[14])
    # # y5
    # particles[:,15] += u[15] + (randn(N) * std[15])
    # # y6
    # particles[:,16] += u[16] + (randn(N) * std[16])
    # # y7
    # particles[:,17] += u[17] + (randn(N) * std[17])
    # # y8
    # particles[:,18] += u[18] + (randn(N) * std[18])
    # # c0
    # particles[:,19] += u[19] + (randn(N) * std[19])
    # # c1
    # particles[:,20] += u[20] + (randn(N) * std[20])

    # return particles


    # N = len(particles)
    # update heading
    # print('heading:')
    # print(particles[:, 2])
    # particles[:, 2] += u[0] + (randn(N) * std[0])
    # particles[:, 2] %= 2 * np.pi
    # # print(particles[:, 2])

    # # move in the (noisy) commanded direction
    # dist = (u[1] * dt) + (randn(N) * std[1])
    # # print('dist:')
    # # print(dist)
    # particles[:, 0] += np.cos(particles[:, 2]) * dist
    # particles[:, 1] += np.sin(particles[:, 2]) * dist
    # print('particles:')
    # print(particles)

# def create_particles(ctrlpts, mean, std):
#     particles=[]


    # return particles

# generating particles (sets of control points) for boundaries' estimation of the left ventricular cavity
# N number of particles
def create_particles(control_points, N):

    particle = []

    control_points = np.array(control_points)

    # print('len ocntro pots: ', len(control_points))
    # control points 0 and 4 define main axis in each of the two segmented images
    main_axis_ini = control_points[4] - control_points[0]
    # main_axis_end = control_points[1][4] - control_points[1][0]

    # axis normalization

    main_axis_ini = main_axis_ini / np.linalg.norm(main_axis_ini)
    # main_axis_end = main_axis_end / np.linalg.norm(main_axis_end)

    # first component ang
    main_axis_polar = rect2pol(main_axis_ini)
    particle.append([main_axis_polar[1]])

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

    # second: control points coordinates
    particle.append(xs_ini.tolist())
    particle.append(ys_ini.tolist())

    # third: origin
    particle.append(control_points[0].tolist())

    # xs_ini = np.reshape(xs_ini, (-1,1))
    # ys_ini = np.reshape(ys_ini, (-1,1))

    # pts_ctl = np.concatenate((xs_ini,ys_ini),axis=1)

    # print('xs', xs_ini)
    # print('ys', ys_ini)
    # print('[xs,ys]', pts_ctl, pts_ctl.shape)

    # particles = np.empty((N, len(control_points), 2))


    # print(particle)
    # particle = list(np.concatenate(particle).flat)
    # transform list of lists into a plain numpy array
    particle = np.concatenate(particle)
    # print(particle)
    particles = np.repeat([particle], N, axis=0)
    # print(len(particles), particles.shape)

    # generating N identical particles
    # particles = N*[particle]
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


def pca_raw_features(feat_frames):

    print('feat_frames:',len(feat_frames))
    for cont, feat_single in enumerate(feat_frames):
        print('frame: ', cont)
        print('feat_single:', len(feat_single), len(feat_single[0]))

        pca.fit(feat_single)
        pca_feat = pca.transform(feat_single)

        mean_values = np.mean(pca_feat, axis=0)
        std_values = np.std(pca_feat, axis=0)

        pca_list.append(pca)
        mean_list.append(mean_values)
        std_list.append(std_values)



####### main function ###########
if __name__== '__main__':

    # print(f"Arguments count: {len(sys.argv)}")
    # for i, arg in enumerate(sys.argv):
    #     print(f"Argument {i:>6}: {arg}")

    num_comp = np.int32(sys.argv[1]) # 10, 15, 20
    num_particles = np.int32(sys.argv[2]) # 50, 100, 150

    print('num_components pca: ', num_comp)
    print('num_particles: ', num_particles)


    filename_controlpts = 'data/controlpts_segmentations'
    filename_pca = 'data/pca_data'
    dir_video = '../Videos/'
    filename_segmentations = '../VolumeTracings.csv'
    filename_features_ctrlpts = 'data/features_ctlpts.csv'
    filename_raw_features = 'data/raw_data_before_pca'
    filename_results = 'data/results'

    # reading list of intensity features
    with open(filename_pca, 'rb') as fp:
        # the frames count started at 1 (first frame)
        print('reading pca data ...')
        [X_train, X_test, names_train, names_test, pca_list, mean_list, std_list] = pickle.load(fp)
        print ('done.')

    # reading list of intensity features
    with open(filename_raw_features, 'rb') as fp:
        # the frames count started at 1 (first frame)
        print('reading raw features before pca ...')
        feat_frames = pickle.load(fp)
        print ('done.')
    # print(len(feat_frames),len(feat_frames[0]), len(feat_frames[0][0]) )


    # print('pca: ', len(pca_list))
    # print('mean_list: ', len(mean_list), len(mean_list[0]))
    # print('std_list: ', len(std_list), len(std_list[0]))

    d={'pca': pca_list, 'mean': mean_list, 'std': std_list}
    df_intensities = pd.DataFrame(data=d)
    time_sec = np.linspace(0,60,num=len(pca_list))
    df_intensities.index = pd.to_datetime(time_sec, unit='s')
    # print(df_intensities.head())

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
    # mean_cp = train_cp.mean()
    # std_cp = train_cp.std()
    # print('mean and std:')
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

    results_ctrlpts=[]
    cont_id = 0
    # [cont_id:cont_id+1]
    # reading each test video
    for id_test, filename_test in enumerate(names_test):

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
        dist_ed = np.linalg.norm(main_axis_ini)
        w =  dist_ed / factor
        print('w:',w)

        # scaling all the components of training data by w; all but ang
        new_mean = mean_cp * w
        new_std = std_cp * w
        new_mean['ang'] = mean_cp['ang']
        new_std['ang'] = std_cp['ang']

        # print(new_mean)
        # print(new_std)

        # # drawing an initial B-Spline curve in a ED frame of a the test data set
        # curve_es = BSpline.Curve()
        # # Set evaluation delta (control the number of curve points)
        # curve_es.delta = 0.02
        # curve_es.order = 3
        # # curve.ctrlpts = parameters.tolist()
        # curve_es.ctrlpts = ctrlpts_test_es
        # curve_es.knotvector = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7]


        # calcualting ratio of increments amount frames using a cosine function
        # alphas = np.linspace(0, 3.14159265, len(selected_frames))
        alphas = np.linspace(0, np.pi, len(selected_frames))
        # print('len(selected_frames): ', len(selected_frames))
        # print(alphas)
        ratios=[]
        for a, b in zip(alphas,alphas[1:]):
            # print(a, b)
            ratios.append(((1-np.cos(b))-(1-np.cos(a))) / 2.0)
        # print(len(ratios))
        # print(ratios)
        # print(np.cumsum(ratios))


        # print(new_mean * ratios[0])

        # plt.plot(ratios)
        # plt.plot(np.cumsum(ratios))
        # plt.show()

        # we used 59.9 instead of 60.0 due to approximation issues
        period = 60 / (len(selected_frames)-1)
        # avoiding approx. issues we applied the next two lines for selection of three decimals without rounding
        a = str(period).split('.')
        b=a[0]+'.'+a[1][0:3]
        # print('period:', len(selected_frames), period, b)
        # print('period:', len(selected_frames), period, round(period,3))
        # df_resampled = df_intensities.resample(str(round(period,3))+'S').bfill()
        df_frames_intensities = df_intensities.resample(b+'S').bfill()
        # print(df_frames_intensities)
        # print('frame 0')
        # print(df_frames_intensities.iloc[0])
        # print('frame 1')
        # print(df_frames_intensities.iloc[1])

        # particles initialization, N number of particles
        N=num_particles

        particles = create_particles(ctrlpts_test_ed, N)
        # print(len(particles), len(particles[0]))
        # pf_control_points = []
        # frame visualization
        # img = np.copy(frame_ini)
        # print('before')
        # print(particles)
        count_resampling=0
        acc=0
        threshold_value = 0
        radius=1.0
        weights = np.ones(N) / N
        for id_frame, frame in enumerate(selected_frames):

            # print('frame: ', id_frame, new_mean['y5'])
            print('id frame: ', id_frame)

            if id_frame == 0:
                radius = np.int32(np.rint(dist_ed / 20.0))
                feat_intensities, threshold_value = function_intensities(ctrlpts_test_ed, frame, id_frame, radius, threshold_value)
                frame_ed = frame
                print (len(feat_intensities), feat_intensities)
                print('median value, radius: ', threshold_value, radius)
            else:
                # print('median value, radius: ', threshold_value, radius)
                # mean and standard deviation between two consecutive frames
                delta_mean = new_mean*ratios[id_frame-1]
                delta_std = new_std*ratios[id_frame-1]
                # print('frame: ', id_frame, mean_values['y5'])
                # acc+=mean_values['y5']
                # ctrlpts_sets = particles_generation(ctrlpts_test_es)
                # print('before')
                # print(particles)
                particles = predict(particles, delta_mean, delta_std)

                ctrlpts_particles = drawing_particles(particles)

                feat_particles=[]
                for ctrlpts in ctrlpts_particles:
                    # print('particles:', radius, threshold_value)
                    feat_intensities, threshold_value = function_intensities(ctrlpts, frame, id_frame, radius, threshold_value)
                    # print(feat_intensities)
                    feat_particles.append(feat_intensities)

                    # img = drawing_curve(frame, ctrlpts)
                    # cv2.imshow('frame',img)
                    # cv2.waitKey(50)

                weights = update(feat_particles, df_frames_intensities.iloc[id_frame], num_comp, weights)

                print('iter neff(weights): ', neff(weights), 2*N/3)
                if neff(weights) < 2*N/3:
                    print('iter, resampling')
                    indexes = systematic_resample(weights)
                    print(indexes)
                    weights = resample_from_index(particles, weights, indexes)
                    count_resampling+=1
                #     assert np.allclose(weights, 1/N)
                mean_particles, var_particles = estimate(particles, weights)
                # print('mean particles', mean_particles)

                mean_ctrlpts = drawing_particles([mean_particles])
                # print('mean ctrlpts')
                print(mean_ctrlpts)

                img = drawing_curve(frame, mean_ctrlpts[0])
                cv2.imshow('frame',img)
                cv2.waitKey(50)
                # xs.append(mu)

        frame_es = frame
        results_ctrlpts.append([frame_ed, frame_es, ctrlpts_test_ed, ctrlpts_test_es, mean_ctrlpts[0], len(selected_frames), dist_ed, threshold_value, count_resampling])

        img = drawing_curve(frame, ctrlpts_test_es)
        cv2.imshow('ES',img)
        cv2.waitKey(1000)

                # print('after')
                # print(particles)
                # curve.ctrlpts = update()
        # print('acc:', acc)
        # print('after')
        # print(particles)

        # for frame, control_points in zip(selected_frames, pf_control_points):
        #     drawing_curve(frame, control_points)

        # cv2.waitKey(0)

    with open(filename_results, 'wb') as fp:
        # the frames count started at 1 (first frame)
        print('saving results ...')
        pickle.dump(results_ctrlpts, fp)
        print ('done.')

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
