import matplotlib.pyplot as plt
from PIL import Image

def show_image(path):
    img = Image.open(path)
    plt.imshow(img)
    plt.axis('off')  # Oculta los ejes
    plt.title("Imagen con PIL")
    plt.show()
