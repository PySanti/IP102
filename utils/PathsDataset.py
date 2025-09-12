from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class PathsDataset(Dataset):
    """
        Dataset creado para wrapear las rutas de las
        imagenes de entrenamiento para el calculo de media y desviacion estandar
    """
    def __init__(self, X) -> None:
        self.x_data = X
    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, idx):
        image = Image.open(self.x_data[idx]).convert('RGB')
        image_arr = np.array(image)
        image.close()
        return image_arr
