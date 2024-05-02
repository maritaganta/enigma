import cv2
import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN

video_path = 'data/sinusoidal_move_phase.mp4'

def read_video(video_path):
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]

    if not cap.isOpened():
        print("ERROR: Could not open video")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))

            cnts, corners, line, rotated, width = pipeline(frame, viz=True)
                
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        else:
            break

    cap.release()
    cv2.destroyAllWindows()



def preprocess(frame, size=(480,480)):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    frame = cv2.bitwise_not(frame)

    w, h = frame.shape
    center = (h//2, w//2)

    top_left_x = center[0] - size[0] // 2
    top_left_y = center[1] - size[1] // 2

    bottom_right_x = top_left_x + size[0]
    bottom_right_y = top_left_y + size[1]

    cropped_frame = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    return cropped_frame


def contouring(frame, ellipse=False):
    contours, _ = cv2.findContours(frame, 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_NONE) 
    if ellipse:
        if len(contours):
            cnt = contours[0]
            cnt_img = frame.copy()
            cnt_img = cv2.cvtColor(cnt_img, cv2.COLOR_GRAY2RGB)

            # TODO: Observe how the eclipse fits well when in line, but not well on curve
            #       The eclipse can tell us where to use optical flow and polynomial
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(cnt_img,ellipse,(0,255,0),2)
            cv2.imshow('Contour Features', cnt_img)
        
    return contours


def objective(x, a, b, c):              # TODO: At some point, create polynomial function that works with coefficients
    # TODO: Evaluate performance of each line
    # https://machinelearningmastery.com/curve-fitting-with-python/
    return a * x + b * x**2 + c


def features_fit_curve(frame, corners):

    try:

        x_values =[frame.shape[0]//2] # adding the image center point as well as an anchor
        y_values = [frame.shape[1]//2]
        min_x = frame.shape[1]  # initialize the min and max x to the other end of the range
        max_x = 0
        for i in corners:
            x,y = i.ravel()
            if x < min_x:
                min_x = x # min x value in the frame
            if x > max_x:
                max_x = x # max x value in the frame
            if x != 0 and y !=0 and x!=479 and y!=479: # TODO: improve into ranges to not get x/y values close to the edges
                x_values.append(x)
                y_values.append(y)

        popt, _ = curve_fit(objective, x_values, y_values)
        a, b, c = popt

        return min_x, max_x, a, b, c
    
    except Exception as e:
        print(e)
        pass


def generate_x_line(min_x, max_x, predict=True):

    margin_x = 0 # adjust according to how further from min x do we start fitting - useful in the case of long curves 

    if predict:
        predict_x = 1 # projecting the line for as many points further

    else:
        predict_x = 0 

    x_line = np.arange(min_x + margin_x, max_x+predict_x, 1)

    return x_line


def line_calculation(x, a, b, c): # TODO: Very bad exception handling here. Tries for arrays and excepts for points without any check
    y = objective(x, a, b, c)
    try:
        line = np.stack((x, y), axis=1)
        return line  
    
    except: # TODO: Looks like I will not be needing this, but great if we could have it encoded nicely. It could be useful when calculating the errors
        return (x,y)


def get_slope(line, step=1): # The step refers to how far in the past are we looking for the slope

    point1 = line[-1]
    point2 = line[-1-step]

    delta_y = point2[1] - point1[1]
    delta_x = point2[0] - point1[0]

    slope = np.arctan2(delta_y, delta_x) # TODO: Account for 0 division - check if arctan documentation does it already 

    return slope


def rotating(frame, slope): # TODO: Add anti-rotation for the width vector

    try:
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(slope), 1.0)
        rotated_image = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))   

        return rotated_image

    except Exception as e:
        print(e)


def detect_corners(frame):

    try:
        corners = cv2.goodFeaturesToTrack(frame,50,0.01,10) # TODO: Experiment with how many corners and min distances I need
        corners = np.int0(corners)
        return corners
    
    except:
        print("Waiting for frame...")
        pass


def closest_node(node, nodes):
    dist2 = np.linalg.norm(nodes - node, axis=1)
    #dist_2 = np.sum((nodes - node)**2, axis = 1)
    return np.argmin(dist2)


def define_verticals(frame, step=0):

    vertical = frame[:, frame.shape[0]//2 + step]

    return vertical


def intersection_points(vertical):

    non_zero_indices = np.nonzero(vertical)[0]

    if len(non_zero_indices) == 0:
        first = None
        last = None
    else:
        first = non_zero_indices[0]
        last = non_zero_indices[-1]
        
    return first, last

def calc_lengths(first, last):

    if first:
        length = abs(first - last)
    else:
        length = 0

    return length



def pipeline(frame, viz=False):

    frame = preprocess(frame)
    cnts = contouring(frame)           # TODO: Gigantic task --> Solve the convex contour problem // During testing on the robot
    corners = detect_corners(frame)

    try:

        min_x, max_x, a, b, c = features_fit_curve(frame, corners)
        x_line = generate_x_line(min_x, max_x)
        line = line_calculation(x_line, a, b, c)    # TODO: I can get curve's error by comparing centerY w/ actual curve value of curve X.
                                                    # TODO: We can estimate accuracy from abs(center_curve_value[1]-centerY)
        slope = get_slope(line)
        rotated = rotating(frame, slope)


        lengths = []
        for i in range(-2, 4, 2):  # TODO: Define how many verticals shall be taken, at the moment I have 3 with a step of 2
            vertical = define_verticals(rotated, step=0)
            first_idx, last_idx = intersection_points(vertical)
            length = calc_lengths(first_idx, last_idx)
            lengths.append(length)

        width = sum(lengths) / len(lengths) # TODO: Create an if-block where the width is checked for within accuracy
    
        
        if viz:

            line_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            for point in line:
                x, y = point
                cv2.circle(line_frame, (int(x), int(y)),1,(255,0,255), 1)

            for i in corners:
                x,y = i.ravel()
                cv2.circle(line_frame,(x,y),1,(30, 255, 255),3)

            cv2.imshow('Playback Video', frame)
            cv2.imshow('Visualisation', line_frame)
            cv2.imshow("Rotation", rotated)

            print(width)

    except:
        print("Waiting for more data...")
        line = None
        rotated = None
        width = 0

    return cnts, corners, line, rotated, width


# read_video(video_path)

