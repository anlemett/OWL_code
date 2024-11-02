import warnings
warnings.filterwarnings('ignore')

import os
#import sys

import numpy as np
import pandas as pd

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
INPUT_DIR = os.path.join(DATA_DIR, "EyeTracking3")
OUTPUT_DIR = os.path.join(DATA_DIR, "EyeTracking4")

metrics_list = ['Saccade', 'Fixation', 'Blink',
                'PupilDiameter', 'LeftPupilDiameter', 'RightPupilDiameter',
                'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
                'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
                'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
                'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
                'HeadHeading', 'HeadPitch',	'HeadRoll']

metrics_sublist = ['PupilDiameter', 'LeftPupilDiameter', 'RightPupilDiameter',
                   'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
                   'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
                   'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
                   'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed']

column_names = ['UnixTimestamp'] + ['SamplePerSecond'] + metrics_list

filenames = ["D1r1", "D1r2", "D1r3",
             "D1r4", "D1r5", "D1r6",
             "D2r1", "D2r2",
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
             "D8r1", "D8r2", "D8r3",
             "D8r4", "D8r5", "D8r6",
             "D9r1", "D9r2", "D9r3",
             "D9r4", "D9r5", "D9r6"
             ]

# for testing
#filenames = ["D3r2"]

for filename in filenames:
    print(filename)
    full_filename = os.path.join(INPUT_DIR, "ET_" + filename +  ".csv")
    df = pd.read_csv(full_filename, sep=' ')
        
    df = df[df['SamplePerSecond']<=250]
    
    first_timestamp = df['UnixTimestamp'].iloc[0]
    last_timestamp = df['UnixTimestamp'].tolist()[-1]
    
    new_df = df.copy()
    
    for ts in range(first_timestamp, last_timestamp + 1):
        
        ts_df = df[df['UnixTimestamp']==ts]
        
        if ts_df.empty:
            print(filename + ": empty second")

            # add 250 rows            
            timestamp_lst = [ts]*250
            sample_per_second_lst = range(1,251)
            metric_values_lst = [np.nan]*250
            
            df_to_add = pd.DataFrame()
            
            df_to_add['UnixTimestamp'] = timestamp_lst
            df_to_add['SamplePerSecond'] = sample_per_second_lst
            for metric in metrics_list:
                df_to_add[metric] = metric_values_lst
            
            new_df = pd.concat([new_df, df_to_add])
            continue
            
        number_of_samples = len(ts_df.index)
        
        if number_of_samples < 250:
            #print(filename + ": adding rows")
            
            num_to_add = 250 - number_of_samples
            # add num_to_add rows
            timestamp_lst = [ts]*num_to_add
            sample_per_second_lst = range(number_of_samples + 1, 251)
            metric_values_lst = [np.nan]*num_to_add
            
            df_to_add = pd.DataFrame()
            
            df_to_add['UnixTimestamp'] = timestamp_lst
            df_to_add['SamplePerSecond'] = sample_per_second_lst
            for metric in metrics_list:
                df_to_add[metric] = metric_values_lst
            
            new_df = pd.concat([new_df, df_to_add])

    new_df.sort_values(['UnixTimestamp', 'SamplePerSecond'], ascending=[True, True], inplace=True)
    new_df.reset_index(drop=True, inplace=True)
    
    number_of_timestamps = last_timestamp - first_timestamp + 1
    number_of_rows1 = number_of_timestamps*250
    number_of_rows2 = len(new_df.index)
    
    #print(number_of_timestamps)
    #print(number_of_rows1)
    #print(number_of_rows2)
    
    #print(new_df.isnull().any().any())
    #nan_count = new_df.isna().sum()
    #print(nan_count)
    
    for col in metrics_sublist:
        new_df[col][new_df[col] < 0] = 0
    
    #print(len(new_df.index))
    
    negative_count = new_df['LeftBlinkOpeningAmplitude'].lt(0).sum()
    print(negative_count)
    
    full_filename = os.path.join(OUTPUT_DIR, "ET_" + filename +  ".csv")
    new_df.to_csv(full_filename, sep=' ', encoding='utf-8', index = False, header = True)
    