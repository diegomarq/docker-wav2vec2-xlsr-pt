#!/usr/bin/python3

### Diego
# Check if the file in base_path corresponds to the file specified in column name 
# python3 check_midia_directory.py cv-valid-test.csv audio midia_kaldi_base

import sys, os
import pandas as pd
import numpy as np

def get_all_files(path):
    return os.listdir(path)

def check_file(base_path, name, all_files):
    
    for f in all_files:
        if f in name:
            return os.path.join(base_path, f)
            
    return np.nan

if __name__ == '__main__':
    file_csv = sys.argv[1]
    column_name = sys.argv[2]
    base_path = sys.argv[3]

    print(f"Reading {file_csv}")

    df = pd.read_csv(file_csv, sep='\t', encoding='utf-8')
    num_rows1 = len(df)

    all_files = get_all_files(base_path)

    df[column_name] = df[column_name].apply(lambda x: check_file(base_path, x, all_files))
    df.dropna(subset=[column_name], inplace=True)
    num_rows2 = len(df)

    print(f"Saving new {file_csv}")
    os.remove(file_csv)

    df.to_csv(file_csv, sep='\t', encoding='utf-8', index=False)

    print(f"Rows: {str(num_rows1)} ==> {str(num_rows2)}")
