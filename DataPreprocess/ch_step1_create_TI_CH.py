import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
CH_DIR = os.path.join(DATA_DIR, "CH0_orig")
OUTPUT_DIR = os.path.join(DATA_DIR, "MLInput")

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
             ["D8r1", "D8r2", "D8r3"],
             ["D8r4", "D8r5", "D8r6"],
             ["D9r1", "D9r2", "D9r3"],
             ["D9r4", "D9r5", "D9r6"]
             ]

ML_df = pd.DataFrame()

atco_num = 0

for atco in filenames:
    
    atco_num = atco_num + 1
        
    atco_df = pd.DataFrame()
    run = 1
    for filename in atco:
        print(filename)
        full_filename = os.path.join(CH_DIR, filename + ".csv")
        scores_df = pd.read_csv(full_filename, sep=' ')
        
        timestamps = scores_df['timestamp'].tolist()
        scores = scores_df['score'].tolist()

        number_of_time_intervals = len(scores)-1 
        
        for ti in range (1, number_of_time_intervals + 1):
            new_row = {'ATCO': atco_num, 'Run': run, 'timeInterval': ti,
                       'score': scores[ti]}

            ML_df = pd.concat([ML_df, pd.DataFrame([new_row])], ignore_index=True)
                
        run = run + 1
        
full_filename = os.path.join(OUTPUT_DIR, "ML_CH.csv")
ML_df.to_csv(full_filename, sep=' ', encoding='utf-8', index = False, header = True)

