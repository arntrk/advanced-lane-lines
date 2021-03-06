import tools as tls
import camera
import numpy as np
import cv2
import matplotlib.pyplot as plt


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def sobel_absolute_scaled(gray):
    # allow images with color depth = one 
    if len(gray.shape) != 2:
        raise TypeError
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx) 
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    return scaled_sobel


def sobel_magnitude(gray, sobel_kernel=3):
    # allow images with color depth = one 
    if len(gray.shape) != 2:
        raise TypeError
        
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculate the magnitude 
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    
    return gradmag

def sobel_direction(gray, sobel_kernel=3): #, thresh=(0, np.pi/2)):
    # allow images with color depth = one 
    if len(gray.shape) != 2:
        raise TypeError
    
    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    
    # Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    
    # calculate the direction of the gradient 
    absgraddir = np.arctan2(abs_sobel_y, abs_sobel_x)
    
    return absgraddir
    
    
def channel_threshold(channel, thresh=(170,255)):
    # Threshold color channel
    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    
    return binary


def find_lane_pixels(binary_warped, leftx_base, rightx_base):
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 150
    # Set minimum number of pixels found to recenter window
    minpix = 4

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low   = leftx_current  - margin 
        win_xleft_high  = leftx_current  + margin 
        win_xright_low  = rightx_current - margin 
        win_xright_high = rightx_current + margin 
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low,  win_y_low), (win_xleft_high,  win_y_high), (0,255,0), 2) 
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #        
        good_left_inds  = ((nonzeroy >= win_y_low)     & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
      

        
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


class lane_detector:
    def __init__(self, filname):
        data = camera.load(filname)
        self.mtx = data["mtx"]
        self.dist = data["dist"]
        self.r_thresh=(145, 200)
        self.g_thresh=(200, 255)
        self.b_thresh=(225,255)
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.color = (255,255,255)
        self.offset = 100
        self.left = []
        self.right = []
        self.prev_left = 0
        self.prev_right = 0
        self.out = 'output_images'
        self.base = 'frame'
        self.ext = '.jpg'
        self.debug_binary = False
        self.debug_lines = False
        self.debug_histogram = False
        self.debug_warped = False
        self.debug_detect = False

    
    def threshold_blue(self, thresh):
        self.b_thresh = tuple(thresh)
        
    def undistort(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
    
    def warp(self, binary, src, dst):
        # only width and height, discard depth if provided
        #w,h = binary.shape[1::-1]
        
        # get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        
        w,h = np.amax(dst,axis=0)

        # returned the warped image
        return cv2.warpPerspective(binary, M, (w,h), flags=cv2.INTER_LINEAR)
    
    def unwarp(self, binary, src, dst, w, h):        
        # get Minv, the transform matrix
        Minv = cv2.getPerspectiveTransform(dst, src)

        # returned the warped image
        return cv2.warpPerspective(binary, Minv, (w,h), flags=cv2.INTER_LINEAR)

    def gaussian_blur(self, img, kernel_size):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def gradients(self, undist):
        # separate components of hls color space
        (_, _, rgb_r) = tls.channels(undist)
        (_, _, lab_b) = tls.channels(tls.bgr2lab(undist))
        (luv_l, _, _) = tls.channels(tls.bgr2luv(undist))
    
        # color space threshold on lab_b 
        r_channel = sobel_magnitude(lab_b)
        r_binary = channel_threshold(r_channel, self.r_thresh)
        
        # color space threshold on luv_l
        g_binary = channel_threshold(luv_l, self.g_thresh)
    
        # color space threshold of rgb_r channel
        b_binary = channel_threshold(rgb_r, self.b_thresh)

        # RGB or BGR
        return np.dstack((b_binary, g_binary, r_binary))
    
    
    def process(self, image):

        # undistorte the image
        undist = self.undistort(image)

        # get width and height,
        w,h = image.shape[1::-1]

        # apply gradients and threshold
        color = self.gradients(undist)

        if self.debug_binary:
            tls.save_image_as(color*255, '{}_binary{}'.format(self.base, self.ext), tls.path_join(self.out, 'binary'))
        
        # combine all channels with bitwise or
        binary =  cv2.bitwise_or(color[:,:,0], cv2.bitwise_or(color[:,:,1], color[:,:,2]))

        # calculate src and dst points
        src = np.array([[570, h//2 + self.offset],      # top left point
                        [710, h//2 + self.offset],      # top right point
                        [w,h],                          # bottom right point
                        [50,h]], dtype = np.float32)    # bottom left point
                          
    
        (tl,tr,br,bl) = src

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
 
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # calculate the destination points
        dst = np.array([[0,0], [maxWidth-1, 0],
                        [maxWidth-1, maxHeight-1],
                        [0, maxHeight-1]], dtype = np.float32)


        if self.debug_lines:            
            lines = np.copy(undist)
            cv2.line(lines, tuple(src[0]), tuple(src[1]), (255,0,0), 5)
            cv2.line(lines, tuple(src[1]), tuple(src[2]), (255,0,0), 5)
            cv2.line(lines, tuple(src[2]), tuple(src[3]), (255,0,0), 5)
            cv2.line(lines, tuple(src[3]), tuple(src[0]), (255,0,0), 5)
            cv2.line(lines, tuple(dst[0]), tuple(dst[1]), (0,255,0), 5)
            cv2.line(lines, tuple(dst[1]), tuple(dst[2]), (0,255,0), 5)
            cv2.line(lines, tuple(dst[2]), tuple(dst[3]), (0,255,0), 5)
            cv2.line(lines, tuple(dst[3]), tuple(dst[0]), (0,255,0), 5)
            tls.save_image_as(lines, '{}_lines{}'.format(self.base,self.ext), tls.path_join(self.out, 'lines'))
        
        # create color warped image for debugging
        if self.debug_warped:
            colorWarped = self.warp(color, src, dst)
            tls.save_image_as(colorWarped*255, '{}_warped{}'.format(self.base,self.ext), tls.path_join(self.out, 'color'))

        # warp the binary image
        warped = self.warp(binary, src, dst)
        
        if self.debug_warped:    
            tls.save_image_as(warped*255, '{}_warped{}'.format(self.base,self.ext), tls.path_join(self.out, 'warped'))
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum((warped[warped.shape[0]//2:,:]), axis=0)
        
        # calculate the midpoint in the histogram
        midpoint = np.int(histogram.shape[0]//2)
        
        # Find the peak of the left and right halves of the histogram 
        left_peak = np.argmax(histogram[:midpoint])
        right_peak = np.argmax(histogram[midpoint:]) + midpoint
        
        if self.debug_histogram:            
            # Plot the result
            fig = plt.figure()
            
            ax1 = fig.add_subplot(1,1,1)
            ax1.plot(range(warped.shape[1]), (warped.shape[0]//2) - histogram, 'red', linewidth=3)
            ax1.imshow((warped[warped.shape[0]//2:,:]))
            path = '{}/histogram/'.format(self.out)
            tls.ensure_path_exists(path)
            fig.savefig('{}{}.png'.format(path,self.base))
            plt.close(fig)
        
        # find lane pixels
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped, left_peak, right_peak)

        if self.debug_detect:
             out_img[lefty, leftx] = [255, 0, 0]
             out_img[righty, rightx] = [0, 0, 255]
             tls.save_image_as(out_img, '{}_output{}'.format(self.base,self.ext), tls.path_join(self.out, 'output'))

        left_fit = self.left
        right_fit = self.right

        # Fit a second order polynomial to each 
        if len(leftx) > 0:
            left_fit =  np.polyfit(lefty,  leftx, 2)
            self.left = left_fit

        if len(rightx) > 0:    
            right_fit = np.polyfit(righty, rightx, 2) 
            self.right = right_fit
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        
        if len(left_fit) > 0:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        
        if len(right_fit) > 0:
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
        # image to draw results on
        output = np.zeros_like(out_img)

        if len(left_fit) > 0 and len(right_fit) > 0:
            # draw green lines between the two curves
            for i in range(left_fitx.shape[0]):
                x1 = left_fitx[i].astype(np.int)
                x2 = right_fitx[i].astype(np.int)
                cv2.line(output, (x1,i),(x2,i), (0,255,0), 5)
        
        # ....
        y_eval = np.max(ploty)
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        left_curverad = self.prev_left
        right_curverad = self.prev_right

        # calculating curvature
        if len(left_fit) > 0: 
            left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])

        if len(right_fit) > 0:
            right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        
        self.prev_left = left_curverad
        self.prev_right = right_curverad


        # get the inverse matrix and inverse transform
        unwarped = self.unwarp(output, src, dst, w, h)
    
        # mix undist image and the unwarped 'detection' image
        result = cv2.addWeighted(undist, 0.9, unwarped, 0.2, 0)
    
        # format text to be added to image
        curvatureText = 'curvature= {:.0f} m {:.0f} m'.format(left_curverad, 
                                                              right_curverad)

        left = left_peak - midpoint
        right = right_peak - midpoint
        diff = right + left
    
        
        # absolute difference
        adiff = np.abs(diff)
        aleft = np.abs(left)
        aright = np.abs(right)
    
        # vehicle lane 3.7m us highway - https://en.wikipedia.org/wiki/Lane
        lane_size = 3.7
        value = adiff * (lane_size / (aleft + aright))
    
        side = 'at'
        if diff > 0:
            side = '{:.02f} m left of'.format(value)
        elif diff < 0:
            side = '{:.02f} m right of'.format(value)
    
        positionTxt = '{} lane center ({}, {}, {}, {})'.format(side, midpoint, left,right, adiff)
        
        # output curvature and lane position as text
        cv2.putText(result, curvatureText,(50, 50), self.font, 2,self.color,2,cv2.LINE_AA)
        cv2.putText(result, positionTxt,  (50,100), self.font, 2,self.color,2,cv2.LINE_AA)
        
        return result

