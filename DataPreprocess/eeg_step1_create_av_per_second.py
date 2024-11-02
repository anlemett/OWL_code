import warnings
warnings.filterwarnings('ignore')

import os

import numpy as np
import pandas as pd
from statistics import mean 

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
INPUT_DIR = os.path.join(DATA_DIR, "EEG0_orig")
OUTPUT_DIR = os.path.join(DATA_DIR, "EEG1")


def getUnixTimestampS(ts_ms):
    return int(ts_ms/1000)

def getValuesPerSecond(timestamps_dict, unix_timestamp):
    return timestamps_dict[unix_timestamp]

filenames = ["D1r1", "D1r2", "D1r3",
             "D1r4", "D1r5", "D1r6",
             "D2r1", "D2r2", "D2r3",
             "D2r4", "D2r5", "D2r6",
             "D3r1", "D3r2", "D3r3",
             "D3r4", "D3r5", "D3r6",
             "D4r1", "D4r2", "D4r3",
             "D4r4", "D4r5", "D4r6",
             "D5r1", "D5r2", "D5r3",
             "D5r4", "D5r5", "D5r6",
             "D6r1", "D6r2", "D6r3",
             "D6r4", "D6r5", "D6r6",
             "D7r1", "D7r2", "D7r3",
             "D7r4", "D7r5", "D7r6",
                     "D8r5", "D8r6",
             "D9r1", "D9r2", "D9r3",
             "D9r4", "D9r5", "D9r6"
             ]

#filenames = ["D1r1", "D1r2", "D1r3"]

for filename in filenames:
    print(filename)
    full_filename = os.path.join(INPUT_DIR, filename +  ".csv")
    df = pd.read_csv(full_filename, sep=';')
    df.sort_values(['calculatedAt'], ascending=[True], inplace=True)
    df.reset_index(inplace=True)

    df['UnixTimestamp'] = df.apply(lambda row: getUnixTimestampS(row['calculatedAt']), axis=1)
    
    first_timestamp = df['UnixTimestamp'].loc[0]
    last_timestamp = df['UnixTimestamp'].loc[len(df.index)-1]
    
    print(first_timestamp)
    print(last_timestamp)
    
    new_df = pd.DataFrame(columns=['UnixTimestamp', 'workload', 'vigilance', 'stress'])

    for ts in range(first_timestamp, last_timestamp + 1):
 
        ts_df = df[df['UnixTimestamp']==ts]
 
        if ts_df.empty:
            print(filename + ": empty second")
            eeg_wl_av = np.nan
            eeg_vig_av = np.nan
            eeg_stress_av = np.nan

        else:
            eeg_wl_av = mean(ts_df['workload'].tolist())
        
            vig_lst = ts_df['vigilance'].tolist()
        
            for i in range(1, len(vig_lst)):
                if vig_lst[i]>100:
                    vig_lst[i] = vig_lst[i-1] 
        
            eeg_vig_av = mean(vig_lst)
        
            eeg_stress_av = mean(ts_df['stress'].tolist())
        
        
        # append the row
        new_row = {'UnixTimestamp': ts, 'workload': eeg_wl_av,
                   'vigilance': eeg_vig_av, 'stress': eeg_stress_av}
        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
 
    full_filename = os.path.join(OUTPUT_DIR, filename +  ".csv")
    new_df.to_csv(full_filename, sep=' ', encoding='utf-8', index = False, header = True)
