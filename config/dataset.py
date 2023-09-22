from torch.utils.data import Dataset,Subset
import torch
import pandas as pd
import numpy as np
from .utils import normalize_df

class SensorDataset(Dataset):
    def __init__(self,data_path:str,sensor_name:str, num_point_per_sample:int, shift:int = 1):
        super(SensorDataset,self).__init__()
        self.shift = shift
        self.width_sample = num_point_per_sample
        self.sensor_name = sensor_name
        self.data = {}
        self.label = {}
        df = pd.read_csv(data_path,index_col=0)
        df, self.dfmax, self.dfmin = normalize_df(df)
        self.field = df.columns
        for f in self.field:
            self.data[f] = torch.tensor(df.loc[:,f].values,dtype=torch.float32)
            self.data[f] = self.transform(self.data[f])
            #將資料與對應的gt配對
            self.label[f] = self.data[f][1:]
            self.data[f] = self.data[f][:-1]


    def __getitem__(self, index):
        data = torch.stack([self.data[f][index] for f in self.field],dim=1)
        label = self.data[self.sensor_name][index].unsqueeze(1)

        return data, label

    def __len__(self):
        return self.data[self.sensor_name].shape[0]
    
    def transform(self,data):
        '''
        建立時序資料
        '''
        output = []
        for i in range((len(data) - self.width_sample)//self.shift):
            start = i*self.shift
            output.append(data[start : (start + self.width_sample)])
        return torch.stack(output)
    
def split_dataset(dataset:Dataset,test_ratio:float= 0.2,seed:int=20230726):
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)
    ndx = torch.randperm(len(dataset),generator=g_cpu).tolist()
    test_ndx = ndx[:int(test_ratio*len(dataset))]
    train_ndx = ndx[int(test_ratio*len(dataset)):]
    testset = Subset(dataset,test_ndx)
    trainset = Subset(dataset,train_ndx)

    return trainset, testset


if __name__ == "__main__":
    ds = SensorDataset("final_data\\FBCB_230520.csv", "PH_out",num_point_per_sample=12,shift=1)
    print(ds.dfmax)
    print(ds.dfmin)
    # print(ds[0][0].shape)
    # print(ds[0][1].shape)

    