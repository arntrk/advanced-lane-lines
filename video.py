import cv2
import argparse
from tools import *
import numpy as np


def video2image(file, out_path):
    
    vid = cv2.VideoCapture(file)

    count = 0
    success = 1

    while success:
        success, image = vid.read()

        if success:
            cv2.imwrite('{1}/frame{0:04d}.jpg'.format(count, out_path), image)
            count += 1



def image2video(out, path1, path2, debug=False):
    # load and sort files from path1 
    files_from_path1 = files_in_dir(path1, True)
    files_from_path1.sort()
    
    # read one image to get size
    test = cv2.imread(files_from_path1[0])
    h,w = test.shape[:2]
    
    files_from_path2 = files_in_dir(path2, True)
    files_from_path2.sort()

    # assume same shape as files from path1
    size = (w*2,h)
        
        
    # prepare a video to write images to
    filename = out + '.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # create file for writing
    video = cv2.VideoWriter(filename, fourcc, 20, size)
    
    
    for file1, file2 in zip(files_from_path1,files_from_path2): 
        if debug:
            print('adding ' + filename_from_path(file1) + ' ' + filename_from_path(file2))   
        # read first image
        image1 = cv2.imread(file1)
        h1,w1 = image1.shape[:2]

        #read second image
        image2 = cv2.imread(file2)
        h2,w2 = image2.shape[:2]

        # merge two images of same size
        if h1==h2 and w1==w2:
            result = np.concatenate((image1, image2), axis=1)
        else:
            raise NotImplementedError

        # write image to video file
        video.write(result)
    
    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Video image extraction tool',
    )

    parser.add_argument('-o', dest='path', action="store", default='', help='path store all images from video')
    parser.add_argument('filename', help='video file to extracted images from')

    args = parser.parse_args()

    path = args.path

    if path == '':
        path,_ = filename_split(filename_from_path(args.filename))
        ensure_path_exists(path_append_trailing_seperator(path))

    print('actracting images from ' + args.filename + ' to path ' + path)
    video2image(args.filename, path)


