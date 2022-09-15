import sys
import cv2
import numpy as np
import pandas as pd

print('read csv file')
data_coord = pd.read_csv('../VolumeTracings.csv')
# print(type(data_coord))

print(data_coord.head())
print(data_coord.tail(4))
print(data_coord.index)
print(data_coord.columns)

# print(data_coord.at[0,'X1'])
# print(data_coord.at[1,'X1'])


# drawing lines
# Create a black image
img = np.zeros((112,112,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px

# first image
cont_rows=0
file_name = data_coord.at[cont_rows,'FileName']
new_file_name = data_coord.at[cont_rows,'FileName']
frame_number = data_coord.at[cont_rows, 'Frame']
new_frame_number = data_coord.at[cont_rows, 'Frame']
print('first: ',file_name)

# while file_name == new_file_name and frame_number == new_frame_number:

#     x1 = round(data_coord.at[cont,'X1'])
#     y1 = round(data_coord.at[cont, 'Y1'])

#     x2 = round(data_coord.at[cont,'X2'])
#     y2 = round(data_coord.at[cont, 'Y2'])

#     cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)
#     cv2.imshow('frame', img)
#     cv2.waitKey(0)


#     cont+=1
#     new_file_name = data_coord.at[cont,'FileName']
#     new_frame_number = data_coord.at[cont,'Frame']


# for index, row_data in data_coord.iterrows():
#     print('index:',index)
#     x1 = round(row_data['X1'])
#     y1 = round(row_data['Y1'])

#     x2 = round(row_data['X2'])
#     y2 = round(row_data['Y2'])

# # x1 = round(data_coord.at[0,'X1'])
# # y1 = round(data_coord.at[0, 'Y1'])

# # x2 = round(data_coord.at[0,'X2'])
# # y2 = round(data_coord.at[0, 'Y2'])

# # print(x1, y1)
# # print(x2, y2)

# cv2.imshow('frame', img)
# cv2.waitKey(0)


# #print (sys.argv)
# # cap = cv2.VideoCapture('../videos/0X789C6979A03B622B.avi')

# cap = cv2.VideoCapture('../Videos/0X100009310A3BD7FC.avi')

cap = cv2.VideoCapture('../Videos/'+file_name)

#cap = cv2.VideoCapture(sys.argv[1])
cont_frames=1
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    print('frame: ',cont_frames)
    # b,g,r = cv2.split(frame)
    # img = np.concatenate((b,g,r),axis=1)
    # cv2.imshow('frame', img)
    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)

    if cont_frames == frame_number:
        img = np.copy(frame)

        while file_name == new_file_name and frame_number == new_frame_number:

            x1 = round(data_coord.at[cont_rows,'X1'])
            y1 = round(data_coord.at[cont_rows, 'Y1'])

            x2 = round(data_coord.at[cont_rows,'X2'])
            y2 = round(data_coord.at[cont_rows, 'Y2'])

            cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)
            cv2.imshow('frame', img)
            cv2.waitKey(0)


            cont_rows+=1
            new_file_name = data_coord.at[cont_rows,'FileName']
            new_frame_number = data_coord.at[cont_rows,'Frame']

        img2 = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
        img3 = cv2.resize(img2,(448,448),interpolation=cv2.INTER_CUBIC)

        img_ref = np.copy(frame)
        img2_ref = cv2.resize(img_ref,(224,224),interpolation=cv2.INTER_CUBIC)
        img3_ref = cv2.resize(img2_ref,(448,448),interpolation=cv2.INTER_CUBIC)


        img_small = np.concatenate((frame,img),axis=1)
        cv2.imshow('frame', img_small)

        img_con = np.concatenate((img3_ref,img3),axis=1)
        cv2.imshow('frame2',img_con)
        cv2.waitKey(0)


        frame_number = new_frame_number

        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)

    #     first_frame = np.copy(frame)
    # if cont == 61:
    #     second_frame = np.copy(frame)
    #     img = np.concatenate((first_frame,second_frame),axis=1)
    #     cv2.imshow('frame', img)
    #     cv2.waitKey(0)
    cont_frames+=1

    # if cv2.waitKey(1) == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
