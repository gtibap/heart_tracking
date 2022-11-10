# particle filter for motion tracking and estimation of the End of Systole left ventricular cavity
import sys
import numpy as np
import cv2
import pickle
import pandas as pd
from geomdl import BSpline
import matplotlib.pyplot as plt

####### main function ###########
if __name__== '__main__':

    num_comp=['10','15','20']
    num_part=['50','100','150']

    data = pd.read_csv('data/df_total.csv')
    # print(data.head())

    df_eval = pd.DataFrame([],columns=['comp','part','mean','std','min','max'])
    

    for id0 in num_comp:
        # df_sub = pd.DataFrame([],columns=['comp','part','mean','std','min','max'])
        list_values=[]
        for id1 in num_part:
            # print('comp part: ', id0, id1)
            subset_1 = data[data['comp']==np.int32(id0)]
            subset_2 = subset_1[subset_1['part']==np.int32(id1)]
            # print(subset_2)

            mean_value = np.around(subset_2['dice'].mean(), decimals=3)
            std_value = np.around(subset_2['dice'].std(), decimals=3)
            min_value = np.around(subset_2['dice'].min(), decimals=3)
            max_value = np.around(subset_2['dice'].max(), decimals=3)
            
            list_values.append([id0, id1, mean_value, std_value, min_value, max_value])
            # print(list_values)
            # , columns=['comp','part','mean','std','min','max']
        df_values = pd.DataFrame(list_values, columns=['comp','part','mean','std','min','max'])
        # print(df_values)
        df_eval= pd.concat([df_eval,df_values])
    
    print(df_eval)
    df_eval.to_csv('data/df_table_report.csv')

