import os
import random 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, Adagrad, SparseAdam
import matplotlib
import matplotlib.pyplot as plt 
import sys
from model import ResNet14
from Loader.superschedule_loader import SuperScheduleDataset
from Loader.sparsematrix_loader import SparseMatrixDataset, collate_fn
import MinkowskiEngine as ME

class SigmoidWeightedMarginRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(SigmoidWeightedMarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, input1, input2, target):
        basic_loss = F.margin_ranking_loss(input1, input2, target, margin=self.margin, reduction='none')
        difference = input1 - input2
        weights = torch.sigmoid(difference)
        weighted_loss = basic_loss * weights
        loss = torch.sum(weighted_loss) / torch.sum(weights)
        return loss

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
  
    set_seed(42)
    
    learning_rate = float(sys.argv[1])
    number_epochs = int(sys.argv[2])
    directory = str(sys.argv[3])
    train = str(sys.argv[4])
    validation = str(sys.argv[5])
    archi = str(sys.argv[6])
    select_optimizer = str(sys.argv[7])
    select_loss = str(sys.argv[8])
    learning_rate_dir = str(sys.argv[9])
    layer_nums = int(sys.argv[10])
    weighted = bool(sys.argv[11])
    finetune = str(sys.argv[12])
    if finetune == "finetune":
      model_path = str(sys.argv[13])
      cpu_model = str(sys.argv[14])
    else:
      model_path = None
      cpu_model = None
    
    #python -u train.py 1e-4 80 /home/chamika2/transfer_learning/ trainfew validation cpu adam ranking  l4 >  /home/chamika2/tf_results/spade_pca/trainfew/l4l4/training.log &
    
    f = open(directory+archi+"trainlog.txt",'a')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = ResNet14(in_channels=1, out_channels=1, D=2, layer_nums=layer_nums, weighted = weighted) # D : 2D Tensor
    net = net.to(device)
    if model_path:
      net.load_state_dict(torch.load(model_path))
      directory_name = directory+archi+"/"+train+"/"+learning_rate_dir+"_"+select_optimizer+"_"+select_loss+"_"+cpu_model
      os.makedirs(directory_name, exist_ok=True)
      os.makedirs(directory_name+"/models", exist_ok=True)
    else:
      directory_name = directory+archi+"/"+train+"/"+learning_rate_dir+"_"+select_optimizer+"_"+select_loss
      os.makedirs(directory_name, exist_ok=True)
      os.makedirs(directory_name+"/models", exist_ok=True)
  
    if select_loss == "ranking":
      criterion = nn.MarginRankingLoss(margin=1)
    elif select_loss == "sigmoid":
      criterion = SigmoidWeightedMarginRankingLoss(margin=1.0)
    
    if select_optimizer == "adam":
      optimizer = Adam(net.parameters(), lr=learning_rate)    
    elif select_optimizer == "adagrad": 
      optimizer = Adagrad(net.parameters(), lr=learning_rate)    
    elif select_optimizer == "sparseadam":  
      optimizer = SparseAdam(net.parameters(), lr=learning_rate)    
    else:
      optimizer = SGD(net.parameters(), lr=learning_rate)    
    
    SparseMatrix_Dataset = SparseMatrixDataset(directory+train+'.txt')
    train_SparseMatrix = torch.utils.data.DataLoader(SparseMatrix_Dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    SparseMatrix_Dataset_Valid = SparseMatrixDataset(directory+validation+'.txt')
    valid_SparseMatrix = torch.utils.data.DataLoader(SparseMatrix_Dataset_Valid, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
    
    patience = 10  # Number of epochs to wait for improvement before stopping
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(number_epochs) :
      # Train
      net.train()
      train_loss = 0
      train_loss_cnt = 0 
      for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(train_SparseMatrix) :
        torch.cuda.empty_cache()
        torch.save(net.state_dict(), directory_name+"/models/scnn.pth")
       
        SuperSchedule_Dataset = SuperScheduleDataset(mtx_names[0],archi) # Get rid of runtime<1000
        train_SuperSchedule = torch.utils.data.DataLoader(SuperSchedule_Dataset, batch_size=32, shuffle=True, num_workers=0)
        shapes = shapes.to(device)
        
        SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)
        for schedule_batchidx, (schedule, runtime) in enumerate(train_SuperSchedule) :
          if (schedule.shape[0] < 2) : break
          schedule, runtime = schedule.to(device), runtime.to(device)
          optimizer.zero_grad()
          query_feature = net.embed_sparse_matrix(SparseMatrix, shapes)
          query_feature = query_feature.expand((schedule.shape[0], query_feature.shape[1]))
          predict = net.forward_after_query(query_feature, schedule)

          #RankingLoss
          iu = torch.triu_indices(predict.shape[0],predict.shape[0],1)
          pred1, pred2 = predict[iu[0]], predict[iu[1]]
          true1, true2 = runtime[iu[0]], runtime[iu[1]]
          sign = (true1-true2).sign()
          loss = criterion(pred1, pred2, sign)
          train_loss += loss.item()
          train_loss_cnt += 1

          loss.backward()
          optimizer.step()
         
          if (sparse_batchidx % 100 == 0 and schedule_batchidx == 0) :
            print("Epoch: ", epoch, ", MTX: ", mtx_names[0], " " , shapes, "(", sparse_batchidx, "), Schedule : ", schedule_batchidx, ", Loss: ", loss.item())
            print("\tPredict : ", predict.detach()[:5,0])
            print("\tGT      : ", runtime.detach()[:5,0])
            print("\tQuery   : ", query_feature.detach()[0,:5])
      
      #Validation
      net.eval()
      with torch.no_grad() :
        valid_loss = 0
        valid_loss_cnt = 0
        for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(valid_SparseMatrix) :
          torch.cuda.empty_cache()
          SuperSchedule_Dataset = SuperScheduleDataset(mtx_names[0],archi) # Get rid of runtime<1000
          valid_SuperSchedule = torch.utils.data.DataLoader(SuperSchedule_Dataset, batch_size=32, shuffle=True, num_workers=0)
          shapes = shapes.to(device)
          
          SparseMatrix = ME.SparseTensor(coordinates=coords, features=features, device=device)
          for schedule_batchidx, (schedule, runtime) in enumerate(valid_SuperSchedule) :
            if (schedule.shape[0] < 6) : break
            schedule, runtime = schedule.to(device), runtime.to(device)
            query_feature = net.embed_sparse_matrix(SparseMatrix, shapes)
            query_feature = query_feature.expand((schedule.shape[0], query_feature.shape[1]))
            predict = net.forward_after_query(query_feature, schedule)

            #HingeRankingLoss
            iu = torch.triu_indices(predict.shape[0],predict.shape[0],1)
            pred1, pred2 = predict[iu[0]], predict[iu[1]]
            true1, true2 = runtime[iu[0]], runtime[iu[1]]
            sign = (true1-true2).sign()
            loss = criterion(pred1, pred2, sign)
            valid_loss += loss.item()
            valid_loss_cnt += 1
           
            if (sparse_batchidx % 100 == 0 and schedule_batchidx == 0) :
              print("ValidEpoch: ", epoch, ", MTX: ", mtx_names[0], " " , shapes, "(", sparse_batchidx, "), Schedule : ", schedule_batchidx, ", Loss: ", loss.item())
              print("\tValidPredict : ", predict.detach()[:5,0])
              print("\tValidGT      : ", runtime.detach()[:5,0])
              print("\tValidQuery   : ", query_feature.detach()[0,:5])
          
      # After validation loss calculation
      valid_loss_avg = valid_loss / valid_loss_cnt
      if valid_loss_avg < best_loss:
          best_loss = valid_loss_avg
          epochs_no_improve = 0
          # Save the best model
          torch.save(net.state_dict(), directory_name+"/best_model.pth")
      else:
          epochs_no_improve += 1
          print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
          if epochs_no_improve >= patience:
              print("Early stopping triggered.")
              early_stop = True
              break  # Break out of the training loop
      
      torch.save(net.state_dict(), directory_name+"/models/"+str(epoch)+"_scnn.pth") 
      print ("--- Epoch {} : Train {} Valid {} ---".format(epoch, train_loss/train_loss_cnt, valid_loss/valid_loss_cnt))
      f.write("--- Epoch {} : Train {} Valid {} ---\n".format(epoch, train_loss/train_loss_cnt, valid_loss/valid_loss_cnt))
      f.flush()
      
    if early_stop:
        print("Training stopped early due to lack of improvement in validation loss.")
    f.close()
    
#cpu: python -u train.py 1e-4 80 /home/chamika2/transfer_learning/ trainfew validation cpu adam ranking l4 no_finetune
#gpu: python -u train.py 1e-4 80 /home/chamika2/transfer_learning/ trainfew validation gpu adam ranking l4 finetune /home/chamika2/transfer_learning/cpu/trainfew/l4_adam_ranking/best_model.pth l4_adam_ranking
#spade: python -u train.py 1e-4 80 /home/chamika2/transfer_learning/ trainfew validation spade adam sigmoid l4 finetune /home/chamika2/transfer_learning/cpu/trainfew/l4_adam_ranking/best_model.pth l4_adam_ranking