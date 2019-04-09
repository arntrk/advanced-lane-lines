import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os, argparse, pickle


import tools as tls


def calibrate(path, nx, ny, save=False, out_path = 'output_images'):
    objpoints = []
    imgpoints = []

    objpnt = np.zeros((ny*nx, 3), np.float32)
    objpnt[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # get all files in directory with path included 
    camera_cal_files = tls.files_in_dir(path, True)

    for file in camera_cal_files:
        # read image from file
        image = cv2.imread(file)

        # convert image to grayscale 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objpnt)
            
            if save == True:
                cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
                head, tail = os.path.split(file)
                tls.save_image_as(image, 'corners_{}'.format(tail), out_path) 

    # returns ret, mtx, dist, rvecs, tvecs
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def store(file, ret, mtx, dist, rvecs, tvecs):
    data = {"ret": ret, "mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}
    dist_pickle = open(file, 'wb')
    pickle.dump(data,  dist_pickle)
    dist_pickle.close()

def load(file):
    dist_pickle_file = open( file, "rb" )
    data = pickle.load(dist_pickle_file)
    dist_pickle_file.close()
    return data
    

def main():
    parser = argparse.ArgumentParser(
        description='Camera calibration tool',
    )

    parser.add_argument('--opath', action="store", default='', help='path store find corners results')
    parser.add_argument('-o', dest='file', action="store", default='camera_calibration_data.p', help='filename to store calibration data')

    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('--ipath', action="store", required=True, help='path to calibration files')
    requiredNamed.add_argument('--numx', action="store", required=True, type=int, help='number of chess corners in x direction')
    requiredNamed.add_argument('--numy',  action="store", required=True, type=int, help='number of chess corners in y direction')

    args = vars(parser.parse_args())

    save = args['opath'] != ''

    ret, mtx, dist, rvecs, tvecs = calibrate(args['ipath'], args['numx'], args['numy'], save, args['opath'])

    store(args['file'], ret, mtx, dist, rvecs, tvecs)

if __name__ == "__main__":
    main()