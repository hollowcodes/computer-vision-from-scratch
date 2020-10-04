
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

from segmentation import segmentation



def main(image_file: str="", k: int=3, iterations: int=5) -> None:
    # load image
    pil_image = Image.open(image_file)
    pil_image = np.array(pil_image)
    original = pil_image

    # create segmentation
    start = time.time()
    segmentation_image = segmentation(pil_image, k, iterations)
    duration = time.time() - start
    print("took:", duration, "seconds.")

    # plot original
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(original)
    title0 = axs[0].title
    title0.set_text("original")
    title0.set_position([.5, 1.17])
    axs[0].axis("off")

    # plot segmentation image
    axs[1].matshow(segmentation_image, cmap=ListedColormap(["y", "b", "g", "purple", "white", "gray", "cyan", "pink", "orange"]))
    title1 = axs[1].title
    title1.set_text("k=" + str(k) + ", iterations= " + str(iterations))
    title1.set_position([.5, 1.0])
    axs[1].axis("off")

    # save fig
    # plt.savefig("images/k2" + str(k) + "_i" + str(iterations) + ".png")
    plt.show()


if __name__ == "__main__":
    image = "images/cern_globe.png"
    main(image_file=image, k=3, iterations=20)
