import os

import cv2
import math
import numpy as np
from utils.image_augmentation import rotate_image, crop_around_center, largest_rotated_rect
from scipy.ndimage import rotate

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
im_in_path = os.path.join(ROOT_DIR, '..', '..', 'dataset', 'negative', '004', 'frame871.jpg')
im_out_path = os.path.join(ROOT_DIR, '..', '..', 'dataset', 't.jpg')



def demo():
    """
    Demos the largest_rotated_rect function
    """

    image = cv2.imread(im_in_path)
    image_height, image_width = image.shape[0:2]
    out_img_hight, out_img_width = 350, 350
    # cv2.imshow("Original Image", image)

    high_diff = (image_height - out_img_hight)/2
    width_diff = (image_width - out_img_width)/2

    image_orig = np.copy(image)
    image_rotated_cropped = rotate(image_orig, 45, reshape=False)
    image_rotated_cropped = image_rotated_cropped[high_diff:high_diff+out_img_hight, width_diff:width_diff+out_img_width]
    # cv2.imshow("cropped", cropped)

    # image_rotated = rotate_image(image, 45)
    # image_rotated_cropped = crop_around_center(
    #     image_rotated,
    #     *largest_rotated_rect(
    #         image_width,
    #         image_height,
    #         math.radians(45)
    #     )
    # )
    # image_rotated_cropped = clipped_zoom(image_rotated_cropped, 1.5)
    cv2.imwrite(im_out_path, image_rotated_cropped)

    #  Work
    #  ==============
    # for i in np.arange(0, 360, 0.5):
    #     image_orig = np.copy(image)
    #     image_rotated = rotate_image(image, i)
    #     image_rotated_cropped = crop_around_center(
    #         image_rotated,
    #         *largest_rotated_rect(
    #             image_width,
    #             image_height,
    #             math.radians(i)
    #         )
    #     )
    #
    #     key = cv2.waitKey(2)
    #     if(key == ord("q") or key == 27):
    #         exit()
    #
    #     cv2.imshow("Original Image", image_orig)
    #     cv2.imshow("Rotated Image", image_rotated)
    #     cv2.imshow("Cropped Image", image_rotated_cropped)

    print "Done"



import numpy as np
from scipy.ndimage import zoom


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]


if __name__ == "__main__":
    demo()