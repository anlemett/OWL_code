import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import math
from statistics import mean
#import sys

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
EEG_DIR = os.path.join(DATA_DIR, "EEG1")
CH_DIR = os.path.join(DATA_DIR, "CH0_orig")
OUTPUT_DIR = os.path.join(DATA_DIR, "EEG2")

TIME_INTERVAL_DURATION = 60  #sec

filenames = [["D1r1", "D1r2", "D1r3"],
             ["D1r4", "D1r5", "D1r6"],
             ["D2r1", "D2r2", "D2r2"],
             ["D2r4", "D2r5", "D2r6"],
             ["D3r1", "D3r2", "D3r3"],
             ["D3r4", "D3r5", "D3r6"],
             ["D4r1", "D4r2", "D4r3"],
             ["D4r4", "D4r5", "D4r6"],
             ["D5r1", "D5r2", "D5r3"],
             ["D5r4", "D5r5", "D5r6"],
             ["D6r1", "D6r2", "D6r3"],
             ["D6r4", "D6r5", "D6r6"],
             ["D7r1", "D7r2", "D7r3"],
             ["D7r4", "D7r5", "D7r6"],
             [],
             [        "D8r5", "D8r6"],
             ["D9r1", "D9r2", "D9r3"],
             ["D9r4", "D9r5", "D9r6"]
             ]


def getTimeInterval(timestamp, ch_first_timestamp, ch_last_timestamp):

    if timestamp < ch_first_timestamp:
        return 0
    if timestamp >= ch_last_timestamp:
        return 0
    return math.trunc((timestamp - ch_first_timestamp)/TIME_INTERVAL_DURATION) + 1


ML_df = pd.DataFrame()

atco_num = 0

for atco in filenames:
    
    atco_num = atco_num + 1
    
    if not atco:
        continue
    
    atco_df = pd.DataFrame()
    run = 1
    for filename in atco:
        print(filename)
        full_filename = os.path.join(EEG_DIR, filename +  ".csv")
        df = pd.read_csv(full_filename, sep=' ')
        
        full_filename = os.path.join(CH_DIR, filename + ".csv")
        scores_df = pd.read_csv(full_filename, sep=' ')
        
        ch_first_timestamp = scores_df['timestamp'].loc[0]
        ch_last_timestamp = scores_df['timestamp'].tolist()[-1]

        df['timeInterval'] = df.apply(lambda row: getTimeInterval(row['UnixTimestamp'],
                                                                  ch_first_timestamp,
                                                                  ch_last_timestamp
                                                                  ),
                                      axis=1) 

        df = df[df['timeInterval']!=0]
        
        eeg_timeintervals = set(df['timeInterval'].tolist())
        number_of_time_intervals = len(eeg_timeintervals)
        
        for ti in range (1, number_of_time_intervals + 1):
            ti_df = df[df['timeInterval']==ti]
            if ti_df.empty or ti_df.dropna().empty:
                 ti_wl_mean = np.nan
                 ti_vig_mean = np.nan
                 ti_stress_mean = np.nan
            else:
                ti_wl_mean = mean(ti_df.dropna()['workload'].tolist())
                ti_vig_mean = mean(ti_df.dropna()['vigilance'].tolist())
                ti_stress_mean = mean(ti_df.dropna()['stress'].tolist())

                
            new_row = {'ATCO': atco_num, 'Run': run, 'timeInterval': ti,
                       'WorkloadMean': ti_wl_mean,
                       'VigilanceMean': ti_vig_mean,
                       'StressMean': ti_stress_mean,
                       }

            ML_df = pd.concat([ML_df, pd.DataFrame([new_row])], ignore_index=True)
                
        run = run + 1

total_nan_count = ML_df.isna().sum().sum()
print("Total number of NaN values in DataFrame: ", total_nan_count)

nan_count = ML_df['WorkloadMean'].isna().sum()
print("Number of NaN values in WorkloadMean:", nan_count)

nan_count = ML_df['VigilanceMean'].isna().sum()
print("Number of NaN values in VigilanceMean:", nan_count)

nan_count = ML_df['StressMean'].isna().sum()
print("Number of NaN values in StressMean:", nan_count)

ML_df = ML_df.dropna()

full_filename = os.path.join(OUTPUT_DIR, "EEG_all_" + str (TIME_INTERVAL_DURATION) + ".csv")
ML_df.to_csv(full_filename, sep=' ', encoding='utf-8', index = False, header = True)
