# Advanced Lane Finding



[image1]: ./output_images/corners_calibration2.jpg
[image2]: ./output_images/corners_calibration3.jpg
[image3]: ./test_images/test1.jpg
[image4]: ./output_images/undist_test1.jpg
[image5]: ./output_images/undist_diff_test1.jpg
[image6]: ./output_images/test_images/binary/test1_binary.jpg
[image7]: ./output_images/test_images/binary/test2_binary.jpg
[warped1]: ./output_images/test_images/warped/test1_warped.jpg
[warped2]: ./output_images/test_images/warped/test2_warped.jpg
[warped3]: ./output_images/test_images/color/test1_warped.jpg
[warped4]: ./output_images/test_images/color/test2_warped.jpg


[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.    

[detailed writeup](writeup.ipynb)


## Running the project:
---

This project have been developed on Linux Ubuntu 18.04

### Prequisits:

Dependency of this project is handled with `pipenv`, and is installed by the following command (Ubuntu 18.04):
```
$ sudo apt-get install pipenv
```

If the following error `ImportError: No named '_tkinter'` when trying to use `matplotlib`, then install `python3-tk` as follows:
```
$ sudo apt-get install python3-tk
```

Install project dependecies and activate the virtualenv, as follows:
```
$ pipenv install
$ pipenv shell
```  

Running jupyter notebook is done as follows:
```
$ jupyter notebook
```

one can also start jupyter notebook from ouside virtualenv, as follows:
```
$ pipenv run jupyter notebook
```


## Computing camera calibration matrix and distorion coefficients


The camera calibration matrix and distortion coefficents are calculated and stored with the `camera.py` tool, which can be called with `-h` or `--help` as parameter for detailed help. For calculating the calibration matrix and distortion coefficents for this project, files in `camera_cal`, is done as follows:  

```
$ python camera.py --ipath camera_cal --numx 9 --numy 6 -o camera_calibration_data.p
```  

The `--opath test_images` parameter can be added to the command above, for generating and storing the result of the corner detection in `test_images` directory.

## Store video as images

The `video.py` script can be used for storing video as images:
```shell
$ python video.py challenge_video.mp4
```
which will put all frames in the video into directory `challenge_video`, since no output path where given as paramter (use `-h` or `--help` help)

**Note**: remeber to active the python environment, see description above.

The same can be done from other python script or jupyter notebook as follows:
```python 
    import video

    video.video2image('challenge_video.mp4', 'challenge_video')
```

There is also a `image2video` function creates a video from images, as follows:
```python 
    import video

    path1 = 'output_images/challenge_video/warped'
    path2 = 'output_images/challenge_video/result'

    video.images2video('challenge_video_analyse.mp4', path1, path2)
```
which will combine `warped` images with `result` images. 
**Note**: doing this requires that the `challenge_video` have been produced and the pipeline have been executed on these images.



## Processing video 

The above example shows how to produce a video by combing two sets of images, so this example will show how to use the pipeline on `challenge_video` output:

```python
    from pipeline import lane_detector
    import tools as tls

    # include True for full path to files
    challenge_files = tls.files_in_dir('challenge_video', True)
    out = 'output_files/challenge_video'

    detect = lane_detector('camera_calibration_data.p')  

    # these are not necessary - but can be used for tweeking threshold values
    detect.r_thresh = ( 90, 255)    # color threshold of S channel in HLS color space
    detect.g_thresh = ( 30, 150)    # absolute x gradient threshold of S channel in HLS color space
    detect.b_thresh = ( 75, 150)    # magnitude gradeient of L channel in HLS color space

    detect.debug_warped = True      # needed to get warped images
    
    for file in challenge_files:
        file_only = tls.filename_from_path(file)
        base, ext = tls.filename_split(file_only)
    
        # need to tell lane_detector which file is being processed
        detect.base = base

        # open file -- full path
        image = cv2.imread(file)
    
        # detect lane lanes
        result = detect.process(image)
    
        # save the result of lane detection
        tls.save_image_as(result, '{}_result{}'.format(base,ext), tls.path_join(out, 'result'))
```

After these one can combine images with `image2video` as shown in example above.

The [P2 notebook](P2.ipynb) is using this pipeline to process video directly using `Moviepy`.

## Issues and Improvements

### Noise due to distance from the car

As the viewing distance increases so will noise relative to the selected threshold values. The resone for this is that the color gets more blury and the lighting condition changes, with respect to distance from camera. One might look at a way to have adaptive threshold values, based on average light condition in the neighborhood of the a pixel.

### Dark and Bright areas

The is issues with both dark areas as under a brigde and bright areas where the sun is shinning on the road or even giving sun flair on the windshield. There will ofcourse also be issues with regards to night driving, were there is very litle color in the image and also bright light from other cars. 

### Curve Fitting

There are times when the lane finding algortihm do not find any points to use for curve fitting, then the np.polyfit will throw error. This might be related to how many parts the domain is splitted up in and there is enough found points to find next placement of search rectangle. 


### TODO:
* Improve by using previous found lane line to get the lane line
* Investiagte adaptive change of threshold values based on L channel in LUV 
* make the `pipeline.py` script process video directly from command-line
* process video stream provided by GStreamer as `udpsrc`, `filesink`, or `device`.
* Implement this on an Jetson Xavier 
 
