import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import math
import statistics
import sys
#from sklearn import preprocessing

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ET_DIR = os.path.join(DATA_DIR, "EyeTracking4")
CH_DIR = os.path.join(DATA_DIR, "CH0_orig")
OUTPUT_DIR = os.path.join(DATA_DIR, "EyeTracking5")

#TIME_INTERVAL_DURATION = 180  #sec
TIME_INTERVAL_DURATION = 60  #sec

filenames = [["D1r1", "D1r2", "D1r3"],
             ["D1r4", "D1r5", "D1r6"],
             ["D2r1", "D2r2"        ],
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
             ["D8r1", "D8r2", "D8r3"],
             ["D8r4", "D8r5", "D8r6"],
             ["D9r1", "D9r2", "D9r3"],
             ["D9r4", "D9r5", "D9r6"]
             ]

#filenames = [["D6r1"]]

new_features = ['SaccadesNumber',
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
                'HeadHeading', 'HeadPitch', 'HeadRoll']

float_columns = ['PupilDiameter', 'LeftPupilDiameter', 'RightPupilDiameter',
                 'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
                 'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
                 'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
                 'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
                 'HeadHeading', 'HeadPitch', 'HeadRoll']



def getTimeInterval(timestamp, ch_first_timestamp, ch_last_timestamp):

    if timestamp < ch_first_timestamp:
        return 0
    if timestamp >= ch_last_timestamp:
        return -1
    return math.trunc((timestamp - ch_first_timestamp)/TIME_INTERVAL_DURATION) + 1


TI_df = pd.DataFrame()

atco_num = 0

for atco in filenames:
    
    atco_num = atco_num + 1
    
    atco_df = pd.DataFrame()
    run = 1
    for filename in atco:
        print(filename)
        full_filename = os.path.join(ET_DIR, 'ET_' + filename +  ".csv")
        df = pd.read_csv(full_filename, sep=' ')
        
        num_rows = len(df)
        print("Number of rows:", num_rows)
        
        total_nan_count = df.isna().sum().sum()
        print("Total number of NaN values in DataFrame before interpolation: ", total_nan_count)
            
        df[float_columns] = df[float_columns].interpolate(method='linear', limit_direction='both', axis=0)
        df[float_columns] = df[float_columns].fillna(method='bfill')
        df[float_columns] = df[float_columns].fillna(method='ffill')

        total_nan_count = df.isna().sum().sum()
        print("Total number of NaN values in DataFrame after interpolation: ", total_nan_count)

        nan_count = df['Saccade'].isna().sum()
        print("Number of NaN values in Saccade before propagation:", nan_count)

        nan_count = df['Fixation'].isna().sum()
        print("Number of NaN values in Fixation before propagation:", nan_count)

        nan_count = df['Blink'].isna().sum()
        print("Number of NaN values in Blink before propagation:", nan_count)        
        
        # Fill NaN values with forward propagation (0 or the number of Saccade/Fixation/Blink)
        # Backward propagation is just for the first rows
        df[['Saccade']] = df[['Saccade']].fillna(method='bfill')
        df[['Saccade']] = df[['Saccade']].fillna(method='ffill')
        df[['Fixation']] = df[['Fixation']].fillna(method='bfill')
        df[['Fixation']] = df[['Fixation']].fillna(method='ffill')
        df[['Blink']] = df[['Blink']].fillna(method='bfill')
        df[['Blink']] = df[['Blink']].fillna(method='ffill')
                
        nan_count = df['Saccade'].isna().sum()
        print("Number of NaN values in Saccade after propagation:", nan_count)

        nan_count = df['Fixation'].isna().sum()
        print("Number of NaN values in Fixation after propagation:", nan_count)

        nan_count = df['Blink'].isna().sum()
        print("Number of NaN values in Blink after propagation:", nan_count)        
 
        #negative_count = df['LeftBlinkOpeningAmplitude'].lt(0).sum()
        #print(negative_count)
                               
        first_timestamp = df['UnixTimestamp'].tolist()[0]
                      
        full_filename = os.path.join(CH_DIR, filename + ".csv")
        scores_df = pd.read_csv(full_filename, sep=' ')
        
        ch_timestamps = scores_df['timestamp'].tolist()
        ch_first_timestamp = ch_timestamps[0]
        
        dif = first_timestamp - ch_first_timestamp
        if dif>0:
            ch_first_timestamp = first_timestamp
            
        number_of_ch_timestamps = len(ch_timestamps)
        ch_last_timestamp = ch_first_timestamp + 180*(number_of_ch_timestamps-1)
        
        df['timeInterval'] = df.apply(lambda row: getTimeInterval(row['UnixTimestamp'],
                                                                  ch_first_timestamp,
                                                                  ch_last_timestamp
                                                                  ),
                                      axis=1) 
                       
        df = df[df['timeInterval']!=0]
        df = df[df['timeInterval']!=-1]
        
        timeIntervals = set(df['timeInterval'].tolist())
        number_of_time_intervals = len(timeIntervals)
                        
        SaccadesNumber = []
        SaccadesDurationMean = []
        SaccadesDurationStd = []
        SaccadesDurationMedian = []
        SaccadesDurationQuantile25 = []
        SaccadesDurationQuantile75 = []
        SaccadesDurationMin = []
        SaccadesDurationMax = []

        FixationDurationMean = []
        FixationDurationStd = []
        FixationDurationMedian = []
        FixationDurationQuantile25 = []
        FixationDurationQuantile75 = []
        FixationDurationMin = []
        FixationDurationMax = []
        
        BlinksNumber = []
        BlinksDurationMean = []
        BlinksDurationStd = []
        BlinksDurationMedian = []
        BlinksDurationQuantile25 = []
        BlinksDurationQuantile75 = []
        BlinksDurationMin = []
        BlinksDurationMax = []

       
        #Add Saccade number, total duration and duration stats per period
        for ti in range(1, number_of_time_intervals+1):
            ti_df = df[df['timeInterval']==ti]
            
            if ti_df.empty:
                continue
            
            ti_df = ti_df.dropna()
            
            if ti_df.empty:
                continue            
            
            ti_saccades_df = ti_df[ti_df['Saccade']!=0]
            if ti_saccades_df.empty:
                print("No Saccade identified for the time interval")
                saccades_number = np.nan
                saccades_duration_mean = np.nan
                saccades_duration_std = np.nan
                saccades_duration_median = np.nan
                saccades_duration_quantile25 = np.nan
                saccades_duration_quantile75 = np.nan
                saccades_duration_min = np.nan
                saccades_duration_max = np.nan

            else:
                saccades_set = set(ti_saccades_df['Saccade'].tolist())
                saccades_number = len(saccades_set)
                saccades_duration = []
                for saccade in saccades_set:
                    saccade_df = ti_df[ti_df['Saccade']==saccade]
                    if not saccade_df.empty:
                        saccades_duration.append(len(saccade_df.index))
                
                saccades_duration_mean = statistics.mean(saccades_duration)
                saccades_duration_std = statistics.stdev(saccades_duration) if len(saccades_duration)>1 else 0
                saccades_duration_median = statistics.median(saccades_duration)
                first_el = saccades_duration[0]
                quantiles = statistics.quantiles(saccades_duration) if len(saccades_duration)>1 else [first_el]*3
                saccades_duration_quantile25 = quantiles[0]
                saccades_duration_quantile75 = quantiles[2]
                saccades_duration_min = min(saccades_duration)
                saccades_duration_max = max(saccades_duration)

            ti_fixation_df = ti_df[ti_df['Fixation']!=0]

            if ti_fixation_df.empty:
                print("No Fixation identified for the time interval")
                fixation_duration_mean = np.nan
                fixation_duration_std = np.nan
                fixation_duration_median = np.nan
                fixation_duration_quantile25 = np.nan
                fixation_duration_quantile75 = np.nan
                fixation_duration_min = np.nan
                fixation_duration_max = np.nan
                
            else:
                fixation_set = set(ti_fixation_df['Fixation'].tolist())
                fixation_duration = []
                for fixation in fixation_set:
                    fixation_df = ti_df[ti_df['Fixation']==fixation]
                    if not fixation_df.empty:
                        fixation_duration.append(len(fixation_df.index))
            
                fixation_duration_mean = statistics.mean(fixation_duration)
                fixation_duration_std = statistics.stdev(fixation_duration) if len(fixation_duration)>1 else 0
                fixation_duration_median = statistics.median(fixation_duration)
                first_el = fixation_duration[0]
                quantiles = statistics.quantiles(fixation_duration) if len(fixation_duration)>1 else [first_el]*3
                fixation_duration_quantile25 = quantiles[0]
                fixation_duration_quantile75 = quantiles[2]
                fixation_duration_min = min(fixation_duration)
                fixation_duration_max = max(fixation_duration)



            ti_blinks_df = ti_df[ti_df['Blink']!=0]
            if ti_blinks_df.empty: # possible if time interval is small
                # set the minimum
                blinks_number = 0
                blinks_duration_mean = 0
                blinks_duration_std = 0
                blinks_duration_median = 0
                blinks_duration_quantile25 = 0
                blinks_duration_quantile75 = 0
                blinks_duration_min = 0
                blinks_duration_max = 0

            else:
                blinks_set = set(ti_blinks_df['Blink'].tolist())
                blinks_number = len(blinks_set)
                blinks_duration = []
                for blink in blinks_set:
                    blink_df = ti_df[ti_df['Blink']==blink]
                    if not blink_df.empty:
                        blinks_duration.append(len(blink_df.index))
                            
                blinks_duration_mean = statistics.mean(blinks_duration)
                blinks_duration_std = statistics.stdev(blinks_duration) if len(blinks_duration)>1 else 0
                blinks_duration_median = statistics.median(blinks_duration)
                first_el = blinks_duration[0]
                quantiles = statistics.quantiles(blinks_duration) if len(blinks_duration)>1 else [first_el]*3
                blinks_duration_quantile25 = quantiles[0]
                blinks_duration_quantile75 = quantiles[2]
                blinks_duration_min = min(blinks_duration)
                blinks_duration_max = max(blinks_duration)
            
            SaccadesNumber.extend([saccades_number]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationMean.extend([saccades_duration_mean]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationStd.extend([saccades_duration_std]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationMedian.extend([saccades_duration_median]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationQuantile25.extend([saccades_duration_quantile25]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationQuantile75.extend([saccades_duration_quantile25]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationMin.extend([saccades_duration_min]*TIME_INTERVAL_DURATION*250)
            SaccadesDurationMax.extend([saccades_duration_max]*TIME_INTERVAL_DURATION*250)
            
            FixationDurationMean.extend([fixation_duration_mean]*TIME_INTERVAL_DURATION*250)
            FixationDurationStd.extend([fixation_duration_std]*TIME_INTERVAL_DURATION*250)
            FixationDurationMedian.extend([fixation_duration_median]*TIME_INTERVAL_DURATION*250)
            FixationDurationQuantile25.extend([fixation_duration_quantile25]*TIME_INTERVAL_DURATION*250)
            FixationDurationQuantile75.extend([fixation_duration_quantile25]*TIME_INTERVAL_DURATION*250)
            FixationDurationMin.extend([fixation_duration_min]*TIME_INTERVAL_DURATION*250)
            FixationDurationMax.extend([fixation_duration_max]*TIME_INTERVAL_DURATION*250)
            
            BlinksNumber.extend([blinks_number]*TIME_INTERVAL_DURATION*250)
            BlinksDurationMean.extend([blinks_duration_mean]*TIME_INTERVAL_DURATION*250)
            BlinksDurationStd.extend([blinks_duration_std]*TIME_INTERVAL_DURATION*250)
            BlinksDurationMedian.extend([blinks_duration_median]*TIME_INTERVAL_DURATION*250)
            BlinksDurationQuantile25.extend([blinks_duration_quantile25]*TIME_INTERVAL_DURATION*250)
            BlinksDurationQuantile75.extend([blinks_duration_quantile25]*TIME_INTERVAL_DURATION*250)
            BlinksDurationMin.extend([blinks_duration_min]*TIME_INTERVAL_DURATION*250)
            BlinksDurationMax.extend([blinks_duration_max]*TIME_INTERVAL_DURATION*250)
        
        
        df['SaccadesNumber'] = SaccadesNumber
        df['SaccadesDurationMean'] = SaccadesDurationMean
        df['SaccadesDurationStd'] = SaccadesDurationStd
        df['SaccadesDurationMedian'] = SaccadesDurationMedian
        df['SaccadesDurationQuantile25'] = SaccadesDurationQuantile25
        df['SaccadesDurationQuantile75'] = SaccadesDurationQuantile75
        df['SaccadesDurationMin'] = SaccadesDurationMin
        df['SaccadesDurationMax'] = SaccadesDurationMax
        
        df['FixationDurationMean'] = FixationDurationMean
        df['FixationDurationStd'] = FixationDurationStd
        df['FixationDurationMedian'] = FixationDurationMedian
        df['FixationDurationQuantile25'] = FixationDurationQuantile25
        df['FixationDurationQuantile75'] = FixationDurationQuantile75
        df['FixationDurationMin'] = FixationDurationMin
        df['FixationDurationMax'] = FixationDurationMax
        
        df['BlinksNumber'] = BlinksNumber
        df['BlinksDurationMean'] = BlinksDurationMean
        df['BlinksDurationStd'] = BlinksDurationStd
        df['BlinksDurationMedian'] = BlinksDurationMedian
        df['BlinksDurationQuantile25'] = BlinksDurationQuantile25
        df['BlinksDurationQuantile75'] = BlinksDurationQuantile75
        df['BlinksDurationMin'] = BlinksDurationMin
        df['BlinksDurationMax'] = BlinksDurationMax

        
        df = df.drop('Saccade', axis=1)
        df = df.drop('Fixation', axis=1)
        df = df.drop('Blink', axis=1)
        
        total_nans = df.isna().sum().sum()
        print("Total NaNs in the DataFrame:", total_nans)
        
        # Fill NaN values: linear interpolation of respective columns
        # (stat. summary features of Saccade, Fixation & Blink)
        # All other columns are processed before (so, no NaNs)
        num_rows = len(df)
        print(f"NUmber of rows before: {num_rows}")
        #df.interpolate(method='linear', limit_direction='both', axis=0, inplace=True)
        # Just for the first and end rows
        #df = df.fillna(method='ffill')
        #df = df.fillna(method='bfill')
        # If NaN values in stat features of Saccade, Fixation or Blink, drop these rows
        df = df.dropna()
        num_rows = len(df)
        print(f"NUmber of rows after: {num_rows}")
        
        row_num = len(df.index)
        #df['ATCO'] = [filename[-2:]] * row_num
        df['ATCO'] = [atco_num] * row_num
        df['Run'] = [run] * row_num
        run = run + 1    

        columns = ['ATCO'] + ['Run'] + ['timeInterval'] + ['UnixTimestamp'] + \
            ['SamplePerSecond'] + new_features
        df = df[columns]
        
        atco_df = pd.concat([atco_df, df], ignore_index=True)
    
    #####################################
    # Normalization per ATCO 
    # might cause data leakage
    '''
    scaler = preprocessing.MinMaxScaler()

    for feature in new_features:
        feature_lst = atco_df[feature].tolist()
        scaled_feature_lst = scaler.fit_transform(np.asarray(feature_lst).reshape(-1, 1))
        atco_df = atco_df.drop(feature, axis = 1)
        atco_df[feature] = scaled_feature_lst
    '''
    #####################################
    
    TI_df = pd.concat([TI_df, atco_df], ignore_index=True)

#print(TI_df.isnull().any().any())
#nan_count = TI_df.isna().sum()
#print(nan_count)

pd.set_option('display.max_columns', None)
#print(TI_df.head(1))

#negative_count = TI_df['LeftBlinkOpeningAmplitude'].lt(0).sum()
#print(negative_count)

full_filename = os.path.join(OUTPUT_DIR, "ET_all_" + str(TIME_INTERVAL_DURATION) + ".csv")
TI_df.to_csv(full_filename, sep=' ', encoding='utf-8', index = False, header = True)
