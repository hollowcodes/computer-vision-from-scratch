from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

from edgeDetection import edgeDetection

def apply_hough_transform(image_path: str=""):
    image = np.array(Image.open(image_path))

    start = time.time()
    edgeImage = edgeDetection(image)
    print("took:", time.time() - start)
    plt.matshow(edgeImage)
    plt.show()


apply_hough_transform(image_path="images/big_house.jpg")
