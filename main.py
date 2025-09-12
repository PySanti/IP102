from utils.ImagesDataset import ImagesDataset
import time
from utils.MACROS import BATCH_SIZE, EPOCHS, MEANS, STDS
from utils.ResNet import ResNet
from utils.load_set import load_set
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np

if __name__ == "__main__":
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

    resnet = ResNet(input_dim=3).cuda()
    optimizer = torch.optim.SGD(resnet.parameters(), lr=5e-4, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    for a in range(EPOCHS):

        epochs_loss = []

        for i, (X_batch, Y_batch) in enumerate(train_loader):
            t1 = time.time()
            X_batch, Y_batch = X_batch.cuda(), Y_batch.cuda()
            optimizer.zero_grad()

            output = resnet(X_batch)
            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()


            epochs_loss.append(loss.item())

            print(f"Batch : {i}/{len(train_loader)}, Time : {time.time()-t1}")
        
        print(f"Epoch : {a+1}, Loss : {np.mean(epochs_loss)}")
