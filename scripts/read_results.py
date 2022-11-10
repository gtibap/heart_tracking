# particle filter for motion tracking and estimation of the End of Systole left ventricular cavity
import sys
import numpy as np
import cv2
import pickle
import pandas as pd
from geomdl import BSpline
import matplotlib.pyplot as plt

def drawing_curve(frame, control_points, resize_flag, color):

    # drawing an initial B-Spline curve in a ED frame of a the test data set
    curve = BSpline.Curve()
    # Set evaluation delta (control the number of curve points)
    curve.delta = 0.02
    curve.order = 3
    # curve.ctrlpts = parameters.tolist()
    curve.ctrlpts = control_points
    curve.knotvector = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7]

    img = np.copy(frame)
    if resize_flag:
        # resizing for a better image visualization
        img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img,(448,448),interpolation=cv2.INTER_CUBIC)
    else:
        pass
    # drawing control points
    # for ctl_pt in curve.ctrlpts:
    #     cv2.circle(img, np.int32(ctl_pt), 3, (0,0,255), -1)
    # drawing the curve
    isClosed=True
    thickness=2
    cv2.polylines(img, np.int32([curve.evalpts]), isClosed, color, thickness)
    # cv2.fillPoly(img,  pts=np.int32([curve.evalpts]), color=color)
    # cv2.fillPoly(img, pts=[points], color=(255, 0, 0))
    # cv2.polylines(img, np.int32([curve_es.evalpts]), 0, (0,255,0), 1)
    # drawing the main axis of the segmented region on the image
    # cv2.line(img, np.int32(curve.ctrlpts[0]), np.int32(curve.ctrlpts[4]), (255,255,0), 1)

    return img

def drawing_area(control_points):

    # drawing an initial B-Spline curve in a ED frame of a the test data set
    curve = BSpline.Curve()
    # Set evaluation delta (control the number of curve points)
    curve.delta = 0.02
    curve.order = 3
    # curve.ctrlpts = parameters.tolist()
    curve.ctrlpts = control_points[0]
    curve.knotvector = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7]

    background = np.zeros((448,448),np.uint8)

    img0 = np.copy(background)
    cv2.fillPoly(img0,  pts=np.int32([curve.evalpts]), color=255)

    #count white pixels
    area_img0 = np.count_nonzero(img0 > 125)

    curve.ctrlpts = control_points[1]
    img1 = np.copy(background)
    cv2.fillPoly(img1,  pts=np.int32([curve.evalpts]), color=255)

    #count white pixels
    area_img1 = np.count_nonzero(img1 > 125)

    curve.ctrlpts = control_points[2]
    img2 = np.copy(background)
    cv2.fillPoly(img2,  pts=np.int32([curve.evalpts]), color=255)

    #count white pixels
    area_img2 = np.count_nonzero(img2 > 125)

    img_result = np.concatenate((img0,img1,img2),axis=1)

    cv2.imshow('particle filter prediction',img_result)
    cv2.waitKey(0)
    # img = np.copy(frame)
    # # resizing for a better image visualization
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
    # img = cv2.resize(img,(448,448),interpolation=cv2.INTER_CUBIC)

    # drawing control points
    # for ctl_pt in curve.ctrlpts:
    #     cv2.circle(img, np.int32(ctl_pt), 3, (0,0,255), -1)
    # drawing the curve
    # isClosed=True
    # thickness=2
    # cv2.polylines(img, np.int32([curve.evalpts]), isClosed, color, thickness)
    # cv2.fillPoly(img,  pts=np.int32([curve.evalpts]), color=color)
    # cv2.fillPoly(img, pts=[points], color=(255, 0, 0))
    # cv2.polylines(img, np.int32([curve_es.evalpts]), 0, (0,255,0), 1)
    # drawing the main axis of the segmented region on the image
    # cv2.line(img, np.int32(curve.ctrlpts[0]), np.int32(curve.ctrlpts[4]), (255,255,0), 1)

    return area_img0, area_img1, area_img2


def visualization(frame_ed, frame_es, ctrlpts_test_ed, ctrlpts_test_es, mean_ctrlpts):

    img0 = drawing_curve(frame_ed, ctrlpts_test_ed, True, (255, 0, 255,))
    # img0 = drawing_curve(img0, ctrlpts_test_es, False, (0, 255,0, 0.2))

    # img1 = drawing_curve(frame_es, ctrlpts_test_ed, True, (255, 0,255))
    img1 = drawing_curve(frame_es, ctrlpts_test_es, True, (0, 255, 0))

    img2 = drawing_curve(frame_es, ctrlpts_test_es, True, (0, 255, 0))
    img2 = drawing_curve(img2, mean_ctrlpts, False, (255, 0, 255))
    # cv2.imshow('frame ED',img)

    img_result = np.concatenate((img0,img1,img2),axis=1)

    cv2.imshow('particle filter prediction',img_result)
    # cv2.imshow('frame ED reference',img0)
    # cv2.imshow('frame ES reference',img1)
    # cv2.imshow('frame ES prediction',img2)
    cv2.waitKey(0)
    return

####### main function ###########
if __name__== '__main__':

    # filename_controlpts = 'data/controlpts_segmentations'
    # filename_feat_inten = 'data/features_intensities'
    # filename_pca = 'data/pca_data'
    # filename_raw_features = 'data/raw_data_before_pca'
    filename_results = 'data/results_10_50'

    # reading list of features
    with open(filename_results, 'rb') as fp:
        # the frames count started at 1 (first frame)
        print('reading features ...')
        results_ctrlpts = pickle.load(fp)
        print ('done.')

    print('results: ', len(results_ctrlpts), len(results_ctrlpts[0]))
    # results_ctrlpts.append([ctrlpts_test_ed, ctrlpts_test_es, mean_ctrlpts[0], len(selected_frames), dist_ed, threshold_value, count_resampling]

    for id, results_video in enumerate(results_ctrlpts):
        [frame_ed, frame_es, ctrlpts_test_ed, ctrlpts_test_es, mean_ctrlpts, num_frames, dist_ed, threshold_value, count_resampling] = results_video
        # print(results_video)
        print(id, num_frames, dist_ed, threshold_value, count_resampling)

        visualization(frame_ed, frame_es, ctrlpts_test_ed, ctrlpts_test_es, mean_ctrlpts)
        # area estimation inside the curves

        # area_ed, area_es, area_pred = drawing_area([ctrlpts_test_ed, ctrlpts_test_es, mean_ctrlpts])
        # print('areas: ', area_ed, area_es, area_pred)







        cv2.destroyAllWindows()
