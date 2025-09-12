from utils.ImagesDataset import ImagesDataset
import time
from utils.MACROS import BATCH_SIZE, EPOCHS, MEANS, STDS
from utils.ResNet import ResNet
from utils.load_set import load_set
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

if __name__ == "__main__":


    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


    print('Cargando rutas de imagenes')
    train_X_paths, train_Y = load_set("./archive/classification","train" )
    val_X_paths, val_Y = load_set("./archive/classification","val" )
    test_X_paths, test_Y = load_set("./archive/classification","test" )

#    means, stds = normalization_metrics_calc(train_X_paths)


    train_transformer  = transforms.Compose([
        transforms.Resize((256, 256)), # redimensionar 
        transforms.CenterCrop((224, 224)), # recordar desde el centro para consistencia
        transforms.ToTensor(),
        transforms.Normalize(MEANS, STDS)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)), # redimensionar 
        transforms.CenterCrop((224, 224)), # recordar desde el centro para consistencia
        transforms.ToTensor(),
        transforms.Normalize(MEANS, STDS)
    ])

    print('Creando dataloaders')
    train_loader = DataLoader(
            ImagesDataset(train_X_paths, train_Y, train_transformer),
            batch_size=BATCH_SIZE, 
            num_workers=12, 
            shuffle=True,
            persistent_workers=True, 
            pin_memory=True) 

    val_loader = DataLoader(
            ImagesDataset(val_X_paths, val_Y, val_transform),
            batch_size=BATCH_SIZE, 
            num_workers=12, 
            shuffle=False,
            persistent_workers=True, 
            pin_memory=True) 



    resnet = ResNet(input_dim=3).to(DEVICE)
    optimizer = torch.optim.SGD(resnet.parameters(), lr=5e-2, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min',patience=8, min_lr=1e-4 )

    for a in range(EPOCHS):

        epochs_train_loss = []
        epochs_val_loss = []

        resnet.train()
        for i, (X_batch, Y_batch) in enumerate(train_loader):
            t1 = time.time()
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            optimizer.zero_grad()

            output = resnet(X_batch)
            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()


            epochs_train_loss.append(loss.item())

            print(f"Batch : {i}/{len(train_loader)}, Time : {time.time()-t1}")



        resnet.eval()

        with torch.no_grad():
            for i, (X_batch, Y_batch) in enumerate(val_loader):
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                output = resnet(X_batch)
                loss = criterion(output, Y_batch)

                epochs_val_loss.append(loss.item())



        scheduler.step(np.mean(epochs_val_loss))        
        print(f"Epoch : {a+1}, Train loss : {np.mean(epochs_train_loss)}, Val loss : {np.mean(epochs_val_loss)}")
