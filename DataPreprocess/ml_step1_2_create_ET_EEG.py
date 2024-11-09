import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import math

#import sys

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ET_DIR = os.path.join(DATA_DIR, "EyeTracking5")
EEG_DIR = os.path.join(DATA_DIR, "EEG2")
ML_DIR = os.path.join(DATA_DIR, "MLInput")

#TIME_INTERVAL_DURATION = 180
#TIME_INTERVAL_DURATION = 60
#TIME_INTERVAL_DURATION = 30
TIME_INTERVAL_DURATION = 10
#TIME_INTERVAL_DURATION = 1

WINDOW_SIZE = 250 * TIME_INTERVAL_DURATION

features = [
            'SaccadesNumber',
            'SaccadesDurationMean', 'SaccadesDurationStd', 'SaccadesDurationMedian',
            'SaccadesDurationQuantile25', 'SaccadesDurationQuantile75',
            'SaccadesDurationMin', 'SaccadesDurationMax',
            'FixationDurationMean', 'FixationDurationStd', 'FixationDurationMedian',
            'FixationDurationQuantile25', 'FixationDurationQuantile75',
            'FixationDurationMin', 'FixationDurationMax',
            'BlinksNumber',
            'BlinksDurationMean', 'BlinksDurationStd', 'BlinksDurationMedian',
            'BlinksDurationQuantile25', 'BlinksDurationQuantile75',
            'BlinksDurationMin', 'BlinksDurationMax',
            'PupilDiameter', 'LeftPupilDiameter', 'RightPupilDiameter',
            'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
            'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
            'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
            'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
            'HeadHeading', 'HeadPitch', 'HeadRoll'
            ]


def get_TS_np(features):
    
    window_size = 250 * TIME_INTERVAL_DURATION
    number_of_features = len(features)
    number_of_features = number_of_features + 1  # + ATCO
    
    # TS_np shape (a,b,c):
    # a - number of time intervals, b - number of measures per time interval (WINDOW_SIZE),
    # c - number of features
    
    # we squeeze to 0 the dimension which we do not know and
    # to which we want to append
    TS_np = np.zeros(shape=(0, window_size, number_of_features))
    
    all_WL_scores = []
    all_Vig_scores = []
    all_Stress_scores = []
    
    #**************************************
    print("Reading Eye Tracking data")
    full_filename = os.path.join(ET_DIR, "ET_all_" + str(TIME_INTERVAL_DURATION) + ".csv")
    et_df = pd.read_csv(full_filename, sep=' ')
    
    #print(et_df.isnull().any().any())
    #The output shows the number of NaN values in each column of the data frame
    #nan_count = et_df.isna().sum()
    #print(nan_count)

    print("Reading EEG data")
    full_filename = os.path.join(EEG_DIR, "EEG_all_" + str(TIME_INTERVAL_DURATION) + ".csv")
    eeg_df = pd.read_csv(full_filename, sep=' ')
    
    
    original_row_count = len(eeg_df)
    #eeg_df = eeg_df.dropna()
    
    rows_dropped = original_row_count - len(eeg_df)
    print(f"Number of eeg_df rows dropped: {rows_dropped}")
  
    dim1_idx = 0
    ti_count = 0

    eeg_ti_empty_count = 0
    eeg_ti_WLMean_NA_count = 0
    for atco_num in range(1,19):

        print(f"ATCO: {atco_num}")

        et_atco_df = et_df[et_df['ATCO']==atco_num]
        eeg_atco_df = eeg_df[eeg_df['ATCO']==atco_num]
        
        if et_atco_df.empty or eeg_atco_df.empty:
            continue
        
        for run in range(1,4):
            et_run_df = et_atco_df[et_atco_df['Run']==run]
            eeg_run_df = eeg_atco_df[eeg_atco_df['Run']==run]
            
            if et_run_df.empty:
                print("et_run_df.empty, continue")
                continue

            if eeg_run_df.empty:
                print("eeg_run_df.empty, continue")
                continue
            
            eeg_run_df.interpolate(method='linear', limit_direction='both', axis=0, inplace=True)
            # Just for the first and end rows
            eeg_run_df = eeg_run_df.fillna(method='ffill')
            eeg_run_df = eeg_run_df.fillna(method='bfill')
            
            number_of_et_time_intervals = max(et_run_df['timeInterval'].tolist())
            print(f"Number of et time intervals: {number_of_et_time_intervals}")
            
            number_of_eeg_time_intervals = max(eeg_run_df['timeInterval'].tolist())
            print(f"Number of eeg time intervals: {number_of_eeg_time_intervals}")
            
            number_of_time_intervals = min(number_of_eeg_time_intervals, number_of_et_time_intervals)
            print(f"Number of time intervals: {number_of_time_intervals}")
            
            run_TS_np = np.zeros(shape=(number_of_time_intervals, window_size, number_of_features))
            run_WL_scores = []
            run_Vig_scores = []
            run_Stress_scores = []
            
            print(f"Number of time intervals: {number_of_time_intervals}")
            dim1_idx = 0
            for ti in range(1, number_of_time_intervals+1):
                et_ti_df = et_run_df[et_run_df['timeInterval']==ti]
                eeg_ti_df = eeg_run_df[eeg_run_df['timeInterval']==ti]
                
                if et_ti_df.empty:
                    print(f"et_ti_df.empty, continue, ti: {ti}")
                    continue
                
                if eeg_ti_df.empty or eeg_ti_df.dropna().empty:
                    eeg_ti_empty_count = eeg_ti_empty_count + 1
                    print(f"eeg_ti_df.empty or eeg_ti_df.dropna().empty, continue, ti: {ti}")
                    continue

                ti_WL_score = eeg_ti_df.iloc[0]['WorkloadMean']
                ti_Vig_score = eeg_ti_df.iloc[0]['VigilanceMean']
                ti_Stress_score =eeg_ti_df.iloc[0]['StressMean']
                
                dim2_idx = 0
                for index, row in et_ti_df.iterrows():
                    #exclude ATCO, Run, timeInterval, UnixTimestamp, SamplePerSecond
                    lst_of_features = row.values.tolist()[5:]
                    run_TS_np[dim1_idx, dim2_idx] = [atco_num] + lst_of_features
                    dim2_idx = dim2_idx + 1
                    
                run_WL_scores.append(ti_WL_score)
                run_Vig_scores.append(ti_Vig_score)
                run_Stress_scores.append(ti_Stress_score)
                        
                dim1_idx = dim1_idx + 1
                
            # dim1_idx - number of time intervals without NaN values
            # > dim1_idx are reserved (to speed up computation) but not used,
            # so remove empty rows
            if dim1_idx < number_of_time_intervals:
                run_TS_np = run_TS_np[:dim1_idx]
                
            run_TS_np_shape = run_TS_np.shape
            print(f"run_TS_np.shape: {run_TS_np_shape}")
            
            ti_count = ti_count + run_TS_np_shape[0]
            print(f"ti_count: {ti_count}")
            
            TS_np = np.append(TS_np, run_TS_np, axis=0)
            
            TS_np_shape = TS_np.shape
            print(f"TS_np.shape: {TS_np_shape}")
            
            all_WL_scores.extend(run_WL_scores)
            all_Vig_scores.extend(run_Vig_scores)
            all_Stress_scores.extend(run_Stress_scores)
            
            
    print(f"eeg_ti_empty_count: {eeg_ti_empty_count}")
    print(f"eeg_ti_WLMean_NA_count: {eeg_ti_WLMean_NA_count}")

    all_scores = np.array((all_WL_scores, all_Vig_scores, all_Stress_scores))
    return (TS_np, all_scores)

(TS_np, scores) = get_TS_np(features)

print(f"Dataframe contains NaNs: {np.isnan(TS_np).any()}")

print(TS_np.shape)

print(len(scores))


# Reshape the 3D array to 2D
TS_np_reshaped = TS_np.reshape(TS_np.shape[0], -1)
print(TS_np_reshaped.shape)

# Save the 2D array to a CSV file
full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__ET.csv")
np.savetxt(full_filename, TS_np_reshaped, delimiter=" ")

# Save scores to a CSV file
full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__EEG.csv")
np.savetxt(full_filename, np.asarray(scores) , delimiter=" ")


