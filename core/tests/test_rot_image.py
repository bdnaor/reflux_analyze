import os

import cv2
import math
import numpy as np
from utils.image_rotate_corp import rotate_image, crop_around_center, largest_rotated_rect

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
im_in_path = os.path.join(ROOT_DIR, '..', '..', 'dataset', 'negative', '004', 'frame871.jpg')
im_out_path = os.path.join(ROOT_DIR, '..', '..', 'dataset', 't.jpg')



def demo():
    """
    Demos the largest_rotated_rect function
    """

    image = cv2.imread(im_in_path)
    image_height, image_width = image.shape[0:2]

    # cv2.imshow("Original Image", image)

    # print "Press [enter] to begin the demo"
    # print "Press [q] or Escape to quit"
    #
    # key = cv2.waitKey(0)
    # if key == ord("q") or key == 27:
    #     exit()

    for i in np.arange(0, 360, 0.5):
        image_orig = np.copy(image)
        image_rotated = rotate_image(image, i)
        image_rotated_cropped = crop_around_center(
            image_rotated,
            *largest_rotated_rect(
                image_width,
                image_height,
                math.radians(i)
            )
        )

        key = cv2.waitKey(2)
        if(key == ord("q") or key == 27):
            exit()

        cv2.imshow("Original Image", image_orig)
        cv2.imshow("Rotated Image", image_rotated)
        cv2.imshow("Cropped Image", image_rotated_cropped)

    print "Done"


if __name__ == "__main__":
    demo()
