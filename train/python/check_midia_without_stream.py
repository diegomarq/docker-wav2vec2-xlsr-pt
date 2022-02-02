#!/usr/bin/python3

### Diego
# Check if midia not contains stream and remove them
# python3 check_midia_without_stream.py cv-valid-test.csv audio midia_kaldi_base

import sys, os
import pandas as pd
import numpy as np

def get_all_files(path):
    return os.listdir(path)

def check_midia(base_path, name, all_files):
    
    for f in all_files:
        if f in name:
            return name if os.path.getsize(name) > 0 else np.nan
            
    return np.nan

if __name__ == '__main__':
    file_csv = sys.argv[1]
    column_name = sys.argv[2]
    base_path = sys.argv[3]

    print(f"Reading {file_csv}")

    df = pd.read_csv(file_csv, sep='\t', encoding='utf-8')
    num_rows1 = len(df)

    all_files = get_all_files(base_path)

    df[column_name].replace('', np.nan, inplace=True)
    df.dropna(subset=[column_name], inplace=True)

    df[column_name] = df[column_name].apply(lambda x: check_midia(base_path, x, all_files))
    df.dropna(subset=[column_name], inplace=True)
    num_rows2 = len(df)

    print(f"Saving new {file_csv}")
    os.remove(file_csv)

    df.to_csv(file_csv, sep='\t', encoding='utf-8', index=False)

    print(f"Rows: {str(num_rows1)} ==> {str(num_rows2)}")
