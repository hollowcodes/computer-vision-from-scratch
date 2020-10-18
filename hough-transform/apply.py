
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

from houghTransform import houghTransform


def apply_hough_transform(image_path: str=""):
    image = np.array(Image.open(image_path))
    lines = houghTransform(image, 10)

    max_px = np.unravel_index(lines.argmax(), lines.shape)
    hough_space_maximum = lines[max_px[0]][max_px[1]]
    rounded_lines = np.around(lines / hough_space_maximum)

    all_thetas, all_ros = [], []
    previous_points = []
    for x in range(len(rounded_lines)):
        for y in range(len(rounded_lines[0])):

            px = rounded_lines[x][y]
            if px == 1:
                
                # try not to save the same line multiple times by checking distances in the parameter space
                same = False
                for point in previous_points:
                    distance = np.sqrt(pow((point[0] - x), 2) + pow((point[1] - y), 2))
                    if distance < 75:
                        same = True
                        break

                if same == False:
                    previous_points.append([x, y])
                    all_ros.append(x)
                    all_thetas.append(y)

    max_distance = int(2 * np.sqrt(pow(image.shape[0], 2) + pow(image.shape[1], 2)))

    # get lines from theta and rho
    all_pts = []
    for i in range(len(all_thetas)):
        ro, theta = all_ros[i], all_thetas[i]

        ro = ro - int(max_distance / 2)
        theta = (theta / 10) * (np.pi / 180)

        a = np.cos(theta)
        b = np.sin(theta)
        x = a * ro 
        y = b * ro

        pt1 = (int(x + 1000*(-b)), int(y + 1000*(a)))
        pt2 = (int(x - 1000*(-b)), int(y - 1000*(a)))

        all_pts.append([pt1, pt2])

    fig, axs = plt.subplots(1, 3)

    axs[0].matshow(rounded_lines)
    axs[1].matshow(lines)

    for pt in all_pts:
        axs[2].plot([pt[0][1], pt[1][1]], [pt[0][0], pt[1][0]], c="r")
        axs[2].imshow(image)
    plt.show()

apply_hough_transform(image_path="images/road_lines2.png")

