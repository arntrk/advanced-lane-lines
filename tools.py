import os
import cv2


def path_join(p1, p2):
    return os.path.join(p1, p2) 

def ensure_path_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def path_append_trailing_seperator(path):
    return path + '/' * (1 - path.endswith('/'))

def files_in_dir(path, include_path=False):
    path_s = path_append_trailing_seperator(path)
    return [path_s * include_path + f for f in os.listdir(path) if os.path.isfile(path_s + f)]

def save_image_as(img, file, path='output_images'):
    path_s = path_append_trailing_seperator(path)
    ensure_path_exists(path_s)
    cv2.imwrite(path_s + file, img) 

def filename_from_path(path):
    return os.path.basename(path)

def filename_split(filename):
    return os.path.splitext(filename)

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def bgr2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def bgr2hls(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

def bgr2lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

def rgb2lab(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

def bgr2luv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

def rgb2luv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def channels(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    return r,g,b
