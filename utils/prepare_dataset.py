import os
import random
from random import randint

import cv2
import math
from PIL import Image
from PIL.Image import LANCZOS
from scipy.ndimage import rotate
from utils.image_augmentation import rotate_image, crop_around_center, largest_rotated_rect

TOTAL_IMAGES_PER_CASE = 2000
image_height = 800
image_width = 800


def reshape_images(input_dataset_path, adaptation_dataset, img_rows, img_cols):
    '''
    This method will create adaptation dataset from the original dataset
    '''

    print 'making dataset'
    # high_diff = (image_height - img_rows)/2
    # width_diff = (image_width - img_cols)/2
    # train_set_size = 0
    folders = os.listdir(input_dataset_path)
    # create output folder
    if not os.path.exists(adaptation_dataset):
        os.makedirs(adaptation_dataset)
    for folder in folders:
        print 'category %s:\n' % folder
        category_path = os.path.join(input_dataset_path, folder)
        sub_folders = os.listdir(category_path)
        for sub_folder in sub_folders:
            case_folder = os.path.join(adaptation_dataset, folder, sub_folder)
            if not os.path.exists(case_folder):
                os.makedirs(case_folder)
            file_list = os.listdir(os.path.join(input_dataset_path, folder, sub_folder))
            # ======================================================================
            #  TODO: fix image augmentation at picture size 50X50
            # number_of_augmentation = 0
            for f in file_list:
                if 'augmentation' in f:
                    os.remove(os.path.join(input_dataset_path, folder, sub_folder, f))
            # ======================================================================
            file_list = os.listdir(os.path.join(input_dataset_path, folder, sub_folder))
            # equalize between amount of frames a cross all patients
            patient_path = os.path.join(input_dataset_path, folder, sub_folder)
            number_of_augmentation = TOTAL_IMAGES_PER_CASE-len(file_list)
            for j in xrange(number_of_augmentation):
                img_index = randint(0, len(file_list)-1)
                image_path_in = os.path.join(patient_path, file_list[img_index])
                image = cv2.imread(image_path_in)
                angel = random.uniform(0.1, 359.9)
                image_rotated = rotate_image(image, angel)
                image_height, image_width = image.shape[0:2]
                image_rotated_cropped = crop_around_center(
                    image_rotated, *largest_rotated_rect(image_width, image_height, math.radians(angel)))
                # image_rotated_cropped = rotate(image, angel, reshape=False)
                # image_rotated_cropped = image_rotated_cropped[high_diff:high_diff + img_rows, width_diff:width_diff + img_cols]

                cv2.imwrite(os.path.join(patient_path, 'augmentation%s.png' % j), image_rotated_cropped)
            file_list = os.listdir(os.path.join(input_dataset_path, folder, sub_folder))
            print 'case %s: frames: %s' % (sub_folder, len(file_list))
            for f in file_list:
                image_path_in = os.path.join(input_dataset_path, folder, sub_folder, f)
                image_path_out = os.path.join(adaptation_dataset, folder, sub_folder, f)
                im = Image.open(image_path_in)
                img = im.resize((img_rows, img_cols), resample=LANCZOS)
                # img = img.convert('L')
                # gray = img.convert('L')d
                # gray.save(output_dtatset+'\\'+f,'JPEG')
                path_without_extention = image_path_out.split('.')[0]
                img.save(path_without_extention, 'PNG')
                # train_set_size += 1
    # end_reshape = time.time()
    # print 'train set size: %s' % (train_set_size,)
    # print 'reshape time:   %s' % (end_reshape - start,)
