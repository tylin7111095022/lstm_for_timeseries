from tqdm import tqdm
import argparse
from config.model import MyLSTM
from config.dataset import SensorDataset, split_dataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=300)
    parser.add_argument("-b", "--batch-size", type=int, default=4096, dest="batch_size")
    parser.add_argument("-w", "--window_length", type=int, default=12)
    parser.add_argument("-l", "--lr", type=float, default=6e-4)
    
    return parser.parse_args()

def main():
    start_time = time.time()
    train_folder = "trainingdata"
    sensor_name = "ORP_in"
    train_files = os.listdir(train_folder)
    hparam = parse_args()
    model = MyLSTM(feature_size=9, hidden_size=9, num_layers = 2, output_size=1)


    optim = torch.optim.AdamW(params=model.parameters(),lr=hparam.lr)
    criteria = torch.nn.MSELoss(reduction="mean")

    for day_data in train_files:
        data_path = os.path.join(train_folder, day_data)
        sensordataset = SensorDataset(data_path, sensor_name, num_point_per_sample=hparam.window_length, shift=1)   
        train(model=model,dataset=sensordataset,optimizer=optim, loss=criteria)

    end_time = time.time()

    print(f"training spent {end_time - start_time} seconds.")

    return
    
def train(model,dataset,optimizer, loss):
    #超參數設定
    hparam = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}.")
    #訓練
    model.to(device)
    model.train()
    trainset, validset = split_dataset(dataset,test_ratio=0.2, seed=20230726)
    print(len(trainset))
    print(len(validset))
    trainsetloader = DataLoader(dataset=trainset,batch_size=hparam.batch_size,shuffle=True)
    validsetloader = DataLoader(dataset=validset,batch_size=hparam.batch_size,shuffle=False)
    history_train_loss = []
    history_valid_loss = []
    iter_epoch = []
    for epoch in range(hparam.epoch):
        print(f"epoch {epoch+1}")
        model.train()
        epoch_train_loss = 0
        epoch_valid_loss = 0
        for batch_id, traindata in enumerate(tqdm(trainsetloader)):             
            x = traindata[0].to(device)
            label = traindata[1].to(device)
            out = model(x)
            batch_loss = loss(out,label)
            epoch_train_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("epoch_train_loss", epoch_train_loss)

        model.eval()
        for batch_id, validdata in enumerate(validsetloader):
            x = validdata[0].to(device)
            label = validdata[1].to(device)
            out = model(x)
            batch_loss = loss(out,label)
            epoch_valid_loss += batch_loss.item()
        print("epoch_valid_loss", epoch_valid_loss)
            
        iter_epoch.append(epoch)
        history_train_loss.append(epoch_train_loss /(len(trainset)) ) #計算mse
        history_valid_loss.append(epoch_valid_loss/(len(validset)) ) #計算mse

        if epoch == 0:
            torch.save(model.state_dict(),f'init.pt')
            minimum_loss = epoch_valid_loss
        if (epoch_valid_loss < minimum_loss):
            minimum_loss = epoch_valid_loss
            print(f"at epoch {epoch+1} best.pt was saved.")
            torch.save(model.state_dict(),f'best.pt')

    # 繪製loss
    plt.plot(iter_epoch,history_train_loss, label = 'train_loss')
    plt.plot(iter_epoch,history_valid_loss, label = 'valid_loss')
    plt.legend()
    plt.xlabel('iters')
    plt.ylabel('Loss')
    plt.show()
    
    return history_train_loss, history_valid_loss

if __name__ == "__main__":
    main()
