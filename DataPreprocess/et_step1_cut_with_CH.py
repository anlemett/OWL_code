import warnings
warnings.filterwarnings('ignore')

import os

import pandas as pd

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
INPUT_DIR = os.path.join(DATA_DIR, "EyeTracking0_orig")
CH_DIR = os.path.join(DATA_DIR, "CH1")
OUTPUT_DIR = os.path.join(DATA_DIR, "EyeTracking1")

metrics_list = ['Saccade', 'Fixation', 'Blink',
                'PupilDiameter', 'LeftPupilDiameter', 'RightPupilDiameter',
                'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
                'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
                'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
                'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
                'HeadHeading', 'HeadPitch',	'HeadRoll']
# for testing:
#metrics_list = ['Fixation', 'LeftPupilDiameter']

# Convert LDAP/Win32 FILETIME to Unix Timestamp
#'realtimeclock' units: 100ns (100ns == 1e-7seconds)
def getUnixTimestamp(file_time):
    winSecs       = int(file_time / 10000000); # divide by 10 000 000 to get seconds
    unixTimestamp = (winSecs - 11644473600); # 1.1.1601 -> 1.1.1970 difference in seconds
    return unixTimestamp

def timeSynch(timestamp, day, run):
    
    #11/21/2023    -13
    #11/24/2023    -10
    #11/28/2023    -8
    #11/29/2023    -7
    #11/30/2023    -6
    #12/5/2023     -1
    #12/6/2023     0
    #12/7/2023     1      until run 44
    #12/7/2023     -24    run 45+
    #12/15/2023    -34
    
    new_timestamp = timestamp
    
    if day==1:
        new_timestamp = new_timestamp - 13
    elif day==2:
        new_timestamp = new_timestamp - 10
    elif day==3:
        new_timestamp = new_timestamp - 8
    elif day==4:
        new_timestamp = new_timestamp - 7
    elif day==5:
        new_timestamp = new_timestamp - 6
    elif day==6:
        new_timestamp = new_timestamp - 1
    elif day==7:
        new_timestamp = new_timestamp - 0
    elif day==8:
        if run < 3:
            new_timestamp = new_timestamp + 1
        else:
            new_timestamp = new_timestamp - 24
    else:
        new_timestamp = new_timestamp - 34

    return new_timestamp


atcos = [
             ["D1r1", "D1r2", "D1r3"],
             ["D1r4", "D1r5", "D1r6"],
             ["D2r1", "D2r2"],
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

for runs in atcos:
    day = int(runs[0][1])
    print(f"Day: {day}")
    for filename in runs:
        print(filename)
        run = int(filename[3])
        print(f"Run: {run}")
        full_filename = os.path.join(INPUT_DIR, filename +  ".log")
        et_df = pd.read_csv(full_filename, sep='\t')

        et_df['UnixTimestamp'] = et_df.apply(lambda row: getUnixTimestamp(row['RealTimeClock']), axis=1)
    
        # adjust ET timestamps (time synchronization)
        et_df['UnixTimestamp'] = et_df.apply(lambda row: timeSynch(row['UnixTimestamp'],
                                                       day,
                                                       run),
                                   axis=1)
    
        full_filename = os.path.join(CH_DIR, filename + ".csv")
        ch_df = pd.read_csv(full_filename, sep=' ')
    
        timestamps = ch_df['timestamp'].tolist()
        ch_first_timestamp = timestamps[0]
        ch_last_timestamp = timestamps[-1]
        
        et_df = et_df[et_df['UnixTimestamp']>=ch_first_timestamp]
        et_df = et_df[et_df['UnixTimestamp']<=ch_last_timestamp]

        columns = ['UnixTimestamp'] + metrics_list
        et_df = et_df[columns]

        full_filename = os.path.join(OUTPUT_DIR, "ET_" + filename +  ".csv")
        et_df.to_csv(full_filename, sep=' ', encoding='utf-8', index = False, header = True)
        
        #num_rows = len(et_df)
        #print("Number of rows:", num_rows)
        
        #both_zero = len(et_df.loc[(et_df['Saccade'] == 0) & (et_df['Fixation'] == 0)])
        #both_non_zero = len(et_df.loc[(et_df['Saccade'] != 0) & (et_df['Fixation'] != 0)])
        #print("Number of rows where both Saccade and Fixation are zero:", both_zero)
        #print("Number of rows where both Saccade and Fixation are non-zero:", both_non_zero)
