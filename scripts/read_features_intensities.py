# feature extraction intensities along the curve
import numpy as np
import cv2
import pickle
import pandas as pd
from geomdl import BSpline
import matplotlib.pyplot as plt
import datetime

def sec2hms(seconds):
    seconds = seconds % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60


    return [np.int32(hours), np.int32(minutes), np.int32(seconds)]

####### main function ###########
if __name__== '__main__':

    filename_controlpts = 'data/controlpts_segmentations'
    filename_feat_inten = 'data/features_intensities'

    # reading list of features
    with open(filename_feat_inten, 'rb') as fp:
        # the frames count started at 1 (first frame)
        print('reading features ...')
        features_intesities_list = pickle.load(fp)
        print ('done.')

    # print(len(features_intesities_list), len(features_intesities_list[0]), len(features_intesities_list[0][0]))

    frames_idx=[]
    frames_numbers = []
    for features_video in features_intesities_list:
        frames_numbers.append(len(features_video))
        # frames_idx.append(np.linspace(0,100,num=len(features_video)))

    print(frames_numbers)

    df_video = pd.DataFrame(features_intesities_list[7])

    # resampling, making all videos with same number of frames
    # arbitrary, we choose 60 s as reference from ED to ES
    time_sec = np.linspace(0,60,num=len(features_intesities_list[7]))
    df_video.index = pd.to_datetime(time_sec, unit='s')
    print(df_video)
    # resampling
    # 18 frames in 60 s, that means 3.529 s between two consecutive frames (60/17)
    print(df_video.resample('3.529S').bfill())




    # time_list=[]
    # for time_id in time_sec:
    #     time_hms = sec2hms(time_id)
    #     print('time: ', time_hms, np.sum(time_hms),pd.to_datetime(np.sum(time_hms), unit='s'))
        
    #     # time_formated = datetime.timedelta(seconds= time_id)
    #     # print(time_formated.hours, time_formated.minutes, time_formated.seconds)
    #     time_list.append(time_hms)
    # print('timelist: ', time_list)
    # # df_video['idx'] = np.linspace(0,10000,num=len(features_intesities_list[0]))
    # df_video.index = pd.to_datetime(time_list) # format='%H:%M:%S'
    # print(df_video)
    # print(frames_numbers)

    


    # print(np.mean(frames_numbers), np.median(frames_numbers))

    # plt.hist(frames_numbers, bins=20)
    # plt.show()

    # dataframe
    
        

