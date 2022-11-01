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

    # reading list of features
    with open(filename_pca, 'rb') as fp:
        # the frames count started at 1 (first frame)
        print('reading pca data ...')
        [X_train, X_test, names_train, names_test, pca_list, mean_list, std_list] = pickle.load(fp)
        print ('done.')
    
    # print(len(X_train), len(X_test), len(pca_list), len(mean_list), len(std_list))
    # print(names_test)

    # read test data set
    ## read file name from a csv file
    print('read csv file')

    data_coord = pd.read_csv('../VolumeTracings.csv')
    # print(data_coord.head())

    seg_data = data_coord[data_coord['FileName']==names_test[0]]
    # print(seg_data)

    selected_frames, frame_ini, frame_end = read_frames(dir_video + names_test[0], seg_data)

    img_ed_es = np.concatenate((frame_ini,frame_end),axis=1)
    cv2.imshow('ref', img_ed_es)
    display_frames(selected_frames)
    # cv2.waitKey(0)

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






