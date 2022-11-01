# feature extraction intensities along the curve
import numpy as np
import cv2
import pickle
import pandas as pd
from geomdl import BSpline
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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
    filename_pca = 'data/pca_data'

    # reading list of features
    with open(filename_feat_inten, 'rb') as fp:
        # the frames count started at 1 (first frame)
        print('reading features ...')
        [filenames_list, features_intesities_list] = pickle.load(fp)
        print ('done.')
    
    # print('filenames:\n', filenames_list)
    # print('feature intensities:\n ', len(features_intesities_list), len(features_intesities_list[0]), len(features_intesities_list[0][0]))
    # print(features_intesities_list[0][0])
    
    # training validation and test splitting
    # From the total videos, 20% for test; from the remining data, 20% for validation and 80% for training 

    X_train, X_test, names_train, names_test = train_test_split(features_intesities_list, filenames_list, test_size=0.2, random_state=42)

    print(len(X_train), len(X_test), len(names_train), len(names_test))

    # train_feat, val_feat, names_tr_feat, names_val_feat = train_test_split(X_train, names_train, test_size=0.2, random_state=42)

    # print(len(train_feat), len(val_feat), len(names_tr_feat), len(names_val_feat))

    # print(len(features_intesities_list), len(features_intesities_list[0]), len(features_intesities_list[0][0]))

    frames_idx=[]
    frames_numbers = []
    dataframe_list=[]
    # training data
    # for features_video in train_feat:
    for features_video in X_train:

        df_video = pd.DataFrame(features_video)
        # resampling, making all videos with same number of frames
        # first, we choose 60s as reference from ED to ES for all videos
        time_sec = np.linspace(0,60,num=len(features_video))
        df_video.index = pd.to_datetime(time_sec, unit='s')
        # resampling to 18 frames (median of frames among the videos)
        # 18 frames in 60s, that means 3.529s between two consecutive frames (60/17)
        # we fill missing values copying the closest one in front of it (back filling)
        df_video = df_video.resample('3.529S').bfill()
        dataframe_list.append(df_video.values.tolist())

        # frames_numbers.append(len(features_video))
        # frames_idx.append(np.linspace(0,100,num=len(features_video)))

    # print('dataframe_list: ', len(dataframe_list), len(dataframe_list[7]), len(dataframe_list[7][0]))

    # initializing a list of lists (18 lists, one list for each frame from ED to ES). Each list will contain features of the same frame, for example ED, from all the videos
    feat_frames = [[] for i in range(len(dataframe_list[0]))]
    # print('feat_frames: ', feat_frames)
    # for elem in feat_frames:
    #     print(id(elem))

    # training data
    for features_video in dataframe_list:
        for id_frame, features_frame in enumerate(features_video):
            feat_frames[id_frame].append(features_frame)
            # print(len(feat_frames))
            # [print(len(elem)) for elem in feat_frames]
            # print(len(feat_frames[0][0]))
            # print('idframe: ', id_frame)
    

    binwidth=0.05
    min=-3.0
    max=3.0
    # PCA to each frame from the training data
    pca = PCA(n_components=25)
    # pca=PCA()
    pca_list=[]
    mean_list=[]
    std_list=[]
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

        # print('mean:', mean_values)
        # print('std:',  std_values)


        # acc = np.cumsum(pca.explained_variance_ratio_)
        # print(cont, pca.singular_values_)
        # print(cont, pca.explained_variance_ratio_)
        # print(cont, acc)
        # # print(feat_single.shape)
        # df = pd.DataFrame(pca_feat)
        # df.hist(bins=np.arange(min, max + binwidth, binwidth))
        # plt.show()

        # df = pd.DataFrame(feat_single)

        # ax = df.hist(column=['ang','x1','x2','x3','x5','x6','x7','y1','y2','y3','y4','y5','y6','y7','c0','c1'], bins=np.arange(min, max + binwidth, binwidth), alpha=0.5)
        # print(df.head)
        # list(range(11, 17))
        # df.hist(column=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24], bins=np.arange(min, max + binwidth, binwidth))
        # df.hist(column=list(range(0,12)), bins=np.arange(min, max + binwidth, binwidth))
        # df.hist(column=list(range(16+16,16+16+16)), bins=np.arange(min, max + binwidth, binwidth))
        
        # df.hist(column=list(range(0,49)), bins=np.arange(min, max + binwidth, binwidth))
        # plt.show()

    with open(filename_pca, 'wb') as fp:
        # the frames count started at 1 (first frame)
        print('saving pca data ...')
        pickle.dump([X_train, X_test, names_train, names_test, pca_list, mean_list, std_list], fp)
        print ('done.')



    # print(dataframe_list[7])
    # print(frames_numbers)
    # print('mean median:', np.mean(frames_numbers), np.median(frames_numbers))
    # plt.hist(frames_numbers)
    # plt.show()

        # df_video = pd.DataFrame(features_intesities_list[7])

        # # resampling, making all videos with same number of frames
        # # arbitrary, we choose 60 s as reference from ED to ES
        # time_sec = np.linspace(0,60,num=len(features_intesities_list[7]))
        # df_video.index = pd.to_datetime(time_sec, unit='s')
    # print(df_video)
    # resampling
    # 18 frames in 60 s, that means 3.529 s between two consecutive frames (60/17)
    # print(df_video.resample('3.529S').bfill())




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
    
        

