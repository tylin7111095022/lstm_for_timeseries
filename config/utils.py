import torch
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

def normalize_df(df:pd.DataFrame):
    min_float = 1e-25 # 防止除以0
    max_dict = np.max(df, axis=0).to_dict()
    min_dict = np.min(df, axis=0).to_dict()
    max_v = np.array(np.max(df, axis=0))
    min_v = np.array(np.min(df, axis=0))
    index = df.index
    col = df.columns
    # print(max_v)
    # print(min_v)
    df = df.to_numpy()
    df = (df - min_v) / (max_v - min_v + min_float)
    new = pd.DataFrame(df,index=index, columns=col)
    return new, max_dict, min_dict

if __name__ == "__main__":
    df = pd.read_csv("final_data\\FBCA_230520.csv",index_col=0)
    print(df.head())
    df = normalize_df(df)
    print(df.head())