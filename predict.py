import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from config.dataset import SensorDataset
from config.model import MyLSTM

def main():
    #宣告參數
    sensor = "ORP_in"
    data_path = "trainingdata\\FBCA_230520.csv"
    window_length = 12
    model = MyLSTM(feature_size=9, hidden_size=9, num_layers = 2, output_size=1)
    model.load_state_dict(torch.load("best.pt"))
    
    #畫圖
    timeseries = pd.read_csv(data_path,index_col=0)[sensor]
    dataset = SensorDataset(data_path, sensor,num_point_per_sample=window_length,shift=1)
    X = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)
    print(f"X: {X.shape}")
    with torch.no_grad():
        # shift predictions for plotting
        model.eval()
        y_pred = model(X)
        y_pred = y_pred[:, -1, :].squeeze()
        y_pred = np.round((y_pred*(dataset.dfmax[sensor]-dataset.dfmin[sensor])+dataset.dfmin[sensor]),decimals=2) #根據dataset 紀錄的最小最大值進行還原
        print(f"y_pred: {y_pred.shape}")
        print(f"timeseries: {timeseries[window_length:-1].shape}")

    fig, ax = plt.subplots(1,1, figsize=(15,8))
    ax.plot(timeseries[window_length:], c='b', label="ground truth")
    ax.set_xticklabels(timeseries.index, rotation = 90)
    ax.plot(y_pred, c='r', label="y_predict")
    ax.legend()
    plt.show()

    #局部window預測
    predict(model=model,
            reference_path=data_path,
            sensor_name=sensor,
            weight_path="best.pt",
            window_length=window_length)

    return 0
    

def predict(model, reference_path:str,sensor_name:str,weight_path="best.pt",window_length=60): 
    """model: 初始化模型物件
       reference_path: 用來預測的csv資料
       sensor_name: 想要預測的特徵欄位
       weight_path: 使用的權重檔
       window_length: 時序資料window的長度"""  
    model.load_state_dict(torch.load(weight_path))
    timeseries = pd.read_csv(reference_path,index_col=0)[sensor_name]
    dataset = SensorDataset(reference_path, sensor_name,num_point_per_sample=window_length,shift=1)
    print(dataset.dfmax)
    print(dataset.dfmin)
    X = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)
    print(f"X: {X.shape}")

    for i in range((len(timeseries)//window_length)+1):
        start = window_length*i
        gt_start = window_length*(i+1)
        with torch.no_grad():
            # shift predictions for plotting
            model.eval()
            gt = timeseries[gt_start:gt_start+window_length]
            x = X[start:start+window_length,:,:] # (B, L, F)
            y_pred = model(x)
            y_pred = y_pred[:, -1, :].squeeze()
            y_pred = np.round((y_pred*(dataset.dfmax[sensor_name]-dataset.dfmin[sensor_name])+dataset.dfmin[sensor_name]),decimals=2) #根據dataset 紀錄的最小最大值進行還原
            print(f"y_pred: {y_pred.shape}")
            print(y_pred)
    
        # plot
        fig, ax = plt.subplots(1,1, figsize=(12,6))
        ax.plot(gt, c='b', label="ground truth")
        ax.set_xticklabels(gt.index, rotation = 90)
        ax.plot(y_pred, c='r', label="y_predict")
        ax.legend()
        plt.show()
    return

if __name__ == "__main__":
    main()