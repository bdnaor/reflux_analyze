import cv2
import os
from PIL import Image


def extract_frames(video_path, frames_path):
    try:
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        success = True
        while success:
            success, image = vidcap.read()
            print('Read a new frame: ', success)
            cv2.imwrite("%s\\frame%d.jpg" % (frames_path, count), image)  # save frame as JPEG file
            count += 1
    except Exception as e:
        print str(e)
        pass


def cut_image(img_src, img_dest):
    #(560,140), (1360, 140) , (560, 940), (1360, 940)
    i = Image.open(img_src)
    frame2 = i.crop(((560, 140, 1360, 940)))
    frame2.save(img_dest)


if __name__ == "__main__":
    video_dir = os.path.join('/home', 'naor', 'Desktop', 'workspace', 'reflux_analyze', 'video')
    img_dir = os.path.join('/home', 'naor', 'Desktop', 'workspace', 'reflux_analyze', 'images')
    # for filename in os.listdir(video_dir):
    #     name, ex = filename.split('.')
    #     if ex.lower() == 'mp4':
    #         out_dir = os.path.join(img_dir, name)
    #         if not os.path.exists(out_dir):
    #             os.makedirs(out_dir)
    #         video_path = os.path.join(video_dir, filename)
    #         extract_frames(video_path, out_dir)
    video_name = '003'
    video_path = os.path.join(video_dir, '%s.MP4' % video_name)
    out_dir = os.path.join(img_dir, video_name)
    extract_frames(video_path, out_dir)

    # for folder_name in os.listdir(img_dir):
    #     for img in os.listdir(os.path.join(img_dir, folder_name)):
    #         try:
    #             img_path = os.path.join(img_dir, folder_name, img)
    #             cut_image(img_path, img_path)
    #         except Exception as e:
    #             print str(e)
    #             pass
