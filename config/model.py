import torch
import torch.nn.functional as F
import torch.nn as nn

class MyLSTM(nn.Module):
    def __init__(self, feature_size:int = 4, hidden_size:int = 8, num_layers:int = 2, output_size:int = 1):
        super(MyLSTM,self).__init__()
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x, _ = self.lstm(x)
        out = self.linear(x)
        return out
    
if __name__ == "__main__":
    model = MyLSTM()
    print(list(model.parameters()))
    # print(model)