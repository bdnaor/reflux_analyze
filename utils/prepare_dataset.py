import os
import random
from random import randint

import cv2
import math
from PIL import Image
from PIL.Image import LANCZOS

from utils.image_rotate_corp import rotate_image, crop_around_center, largest_rotated_rect

TOTAL_IMAGES = 2500


def reshape_images(input_dataset_path, adaptation_dataset, img_rows, img_cols):
    '''
    This method will create adaptation dataset from the original dataset
    '''
    print 'making dataset'
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
            # equalize between amount of frames a cross all patients
            patient_path = os.path.join(input_dataset_path, folder, sub_folder)
            for j in xrange(TOTAL_IMAGES-len(file_list)):
                img_index = randint(0, len(file_list)-1)

                image_path_in = os.path.join(patient_path, file_list[img_index])
                image = cv2.imread(image_path_in)
                angel = random.uniform(0.1, 359.9)
                image_rotated = rotate_image(image, angel)
                image_rotated_cropped = crop_around_center(
                    image_rotated, *largest_rotated_rect(img_cols, img_rows, math.radians(angel)))
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
                img.save(image_path_out, 'JPEG')
                # train_set_size += 1
    # end_reshape = time.time()
    # print 'train set size: %s' % (train_set_size,)
    # print 'reshape time:   %s' % (end_reshape - start,)
