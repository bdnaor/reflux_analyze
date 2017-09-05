import os
import time
from PIL.Image import LANCZOS


def reshape_images(input_dataset_path, adaptation_dtatset, img_rows, img_cols):
    '''
    This method will create adaptation dataset from the original dataset
    '''
    start = time.time()
    train_set_size = 0
    folders = os.listdir(input_dataset_path)
    # create output folder
    if not os.path.exists(adaptation_dtatset):
        os.makedirs(adaptation_dtatset)
    for folder in folders:
        category_path = os.path.join(input_dataset_path, folder)
        sub_folders = os.listdir(category_path)
        for sub_folder in sub_folders:
            case_folder = os.path.join(adaptation_dtatset, folder, sub_folder)
            if not os.path.exists(case_folder):
                os.makedirs(case_folder)
            filelist = os.listdir(case_folder)
            for f in filelist:
                image_path_in = os.path.join(input_dataset_path, folder, sub_folder, f)
                image_path_out = os.path.join(adaptation_dtatset, folder, sub_folder, f)
                im = Image.open(image_path_in)
                img = im.resize((img_rows, img_cols), resample=LANCZOS)
                # img = img.convert('L')
                # gray = img.convert('L')d
                # gray.save(output_dtatset+'\\'+f,'JPEG')
                img.save(image_path_out, 'JPEG')
                train_set_size += 1
    end_reshape = time.time()
    print 'train set size: %s' % (train_set_size,)
    print 'reshape time:   %s' % (end_reshape - start,)
