import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ET_DIR = os.path.join(DATA_DIR, "EyeTracking5")
ML_DIR = os.path.join(DATA_DIR, "MLInput")

WINDOW_SIZE = 250 * 180

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
    
    window_size = 250 * 180
    number_of_features = len(features)
    number_of_features = number_of_features + 1  # + ATCO
    
    # TS_np shape (a,b,c):
    # a - number of time intervals, b - number of measures per time interval (WINDOW_SIZE),
    # c - number of features
    
    # we squeeze to 0 the dimension which we do not know and
    # to which we want to append
    TS_np = np.zeros(shape=(0, window_size, number_of_features))
    all_scores = []
    
    #**************************************
    print("Reading Eye Tracking data")
    full_filename = os.path.join(ET_DIR, "ET_all_180.csv")
    et_df = pd.read_csv(full_filename, sep=' ')
    
    print("Reading CH data")
    full_filename = os.path.join(ML_DIR, "CH_all.csv")
    ch_df = pd.read_csv(full_filename, sep=' ')
    
    dim1_idx = 0

    for atco_num in range(1,19):
        print(f"ATCO: {atco_num}")

        et_atco_df = et_df[et_df['ATCO']==atco_num]
        ch_atco_df = ch_df[ch_df['ATCO']==atco_num]
                
        if et_atco_df.empty:
            continue
        
        for run in range(1,4):
            et_run_df = et_atco_df[et_atco_df['Run']==run]
            ch_run_df = ch_atco_df[ch_atco_df['Run']==run]
            
            if et_run_df.empty:
                continue
        
            number_of_time_intervals = len(ch_run_df.index)
        
            run_TS_np = np.zeros(shape=(number_of_time_intervals, window_size, number_of_features))
            print(run_TS_np.shape)
            all_run_scores = ch_run_df['score'].tolist() 
            run_scores = []
                        
            print(number_of_time_intervals)
            dim1_idx = 0
            for ti in range(1, number_of_time_intervals+1):
                et_ti_df = et_run_df[et_run_df['timeInterval']==ti]
                                               
                if et_ti_df.empty or all_run_scores[ti-1]==0:
                    continue
                       
                dim2_idx = 0
                for index, row in et_ti_df.iterrows():
                    #exclude ATCO, Run, timeInterval, UnixTimestamp, SamplePerSecond
                    lst_of_features = row.values.tolist()[5:]
                    run_TS_np[dim1_idx, dim2_idx] = [atco_num] + lst_of_features
                    dim2_idx = dim2_idx + 1
                 
                dim1_idx = dim1_idx + 1
                run_score = all_run_scores[ti-1]
                run_scores.extend([run_score])
                
            if dim1_idx < number_of_time_intervals:
                run_TS_np = run_TS_np[:dim1_idx]
                
            TS_np = np.append(TS_np, run_TS_np, axis=0)
            all_scores.extend(run_scores)

    return (TS_np, all_scores)

(TS_np, scores) = get_TS_np(features)

print(TS_np.shape) # (667, 45000, 38)
print(len(scores))

# Reshape the 3D array to 2D
TS_np_reshaped = TS_np.reshape(TS_np.shape[0], -1)
# Save the 2D array to a CSV file
full_filename = os.path.join(ML_DIR, "ML_ET_CH__ET.csv")
np.savetxt(full_filename, TS_np_reshaped, delimiter=" ")

# Save scores to a CSV file
full_filename = os.path.join(ML_DIR, "ML_ET_CH__CH.csv")
np.savetxt(full_filename, np.asarray(scores) , fmt='%i', delimiter=" ")
