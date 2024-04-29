import cv2
import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN



video_path = 'data/sinusoidal_move_phase.mp4'

def read_video(video_path, q3d=False, optical_flow=False):
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Could not open video")
        return
    
    prev_frame = None
    current_frame = None
    
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:

            if current_frame is not None:
                prev_frame = current_frame
            current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.circle(frame, (frame.shape[1]//2, frame.shape[0]//2), 1, (0,0,255), 2)
            cv2.imshow('Playback Video', frame)

            if q3d:
                frame = preprocess(current_frame)
                #cnts = contouring(frame)
                corners = detect_corners(frame)

                try:

                    min_x, max_x, a, b, c = features_fit_curve(frame, corners)
                    x_line = generate_x_line(min_x, max_x)
                    line = line_calculation(x_line, a, b, c)

                    width = dist_calc(corners)

                    line_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                    for point in line:
                        x, y = point
                        cv2.circle(line_frame, (int(x), int(y)),1,(255,0,255), 1)

                    for i in corners:
                        x,y = i.ravel()
                        cv2.circle(line_frame,(x,y),3,(30, 255, 255),-1)
                    
                    # IN CASE I NEED CONVEX
                    # hull = cv2.convexHull(corners, False)
                    # hulls = [hull]
                    # cv2.drawContours(line_frame, hulls, 0, (200,0,200), 1, 8)

                    cv2.imshow('Contour Overlay', line_frame)

                except Exception as e: 
                    print(e)
                    #pass
                    





            if optical_flow:
                if prev_frame is not None:
                    flow_img, move, core_flow = Opticalflow.analyse(prev_frame, current_frame)
                    width_rgb, width = measure_width(move, core_flow, prev_frame)


            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def preprocess(frame, size=(480,480)):
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


def objective(x, a, b, c):
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
        print("Waiting for features...")
        print(e)
        pass

def generate_x_line(min_x, max_x):

    margin_x = 0 # adjust according to how further from min x do we start fitting - useful in the case of long curves 

    x_line = np.arange(min_x + margin_x, max_x, 1)

    return x_line

def line_calculation(x, a, b, c):
    
    y = objective(x, a, b, c)

    line = np.stack((x, y), axis=1)

    return line  

def contour_fit_curve(frame, cnts):

    if len(cnts)>0:
        cnt = cnts[0]

        x_values =[frame.shape[0]//2] # adding the image center point as well as an anchor
        y_values = [frame.shape[1]//2]
        min_x = frame.shape[1]  # initialize the min and max x to the other end of the range
        max_x = 0
        for point in cnt:
            x = point[0][0]
            y = point[0][1]
            if x < min_x:
                min_x = x # min x value in the frame
            if x > max_x:
                max_x = x # max x value in the frame
            if x != 0 and y !=0 and x!=479 and y!=479: # TODO: improve into ranges to not get x/y values close to the edges
                x_values.append(x)
                y_values.append(y)

        popt, _ = curve_fit(objective, x_values, y_values)
        a, b, c = popt

        margin_x = 0 # adjust according to how further from min x do we start fitting - useful in the case of long curves 

        x_line = np.arange(min_x + margin_x, max_x, 1)
        y_line = objective(x_line, a, b, c)

        line = np.stack((x_line, y_line), axis=1)

        return line
    

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


def dist_calc(corners):

    # TODO: Remove the hardcoding of the corner idx - used closest node function above, but didn't work...

    try: 
        north = corners[0][0]
        south = corners[1][0]
        return np.linalg.norm(north - south)
    
    except:
        print("unable to load corners")
        pass













class Opticalflow:
    def draw_flow(rgb_prev, rgb_nxt, viz=False):
        THRESHOLD_DETECT = 20
        THRESHOLD_IGNORE_L = 1
        THRESHOLD_IGNORE_H = 300
        MOVE = [0, 0]
        COUNT_MOVE = 1
        STEP = 20

        h, w = rgb_prev.shape[:2]
        y, x = np.mgrid[STEP / 2 : h : STEP, STEP / 2 : w : STEP].reshape(2, -1).astype(int)
        fx, fy = rgb_nxt[y, x].T
        angle = np.arctan2(fy, fx)

        cluster_lables = DBSCAN(eps=0.01, min_samples=2).fit(angle.reshape(-1, 1))
        core_cluster = angle[cluster_lables != -1]
        core_flow = np.mean(core_cluster, axis=1)

        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)

        vx = vy = 0
        for i in range(len(lines)):
            vect_x = lines[i][1][0] - lines[i][0][0]
            vect_y = lines[i][1][1] - lines[i][0][1]
            if (
                THRESHOLD_IGNORE_L <= abs(vect_x) <= THRESHOLD_IGNORE_H
                or THRESHOLD_IGNORE_L <= abs(vect_y) <= THRESHOLD_IGNORE_H
            ):
                vx += vect_x
                vy += vect_y
                COUNT_MOVE += 1
        MOVE[0] = vx
        MOVE[1] = vy

        if not abs(MOVE[0]) >= THRESHOLD_DETECT or not abs(MOVE[1]) >= THRESHOLD_DETECT:
            MOVE = [0, 0]

        if viz:
            vis = cv2.cvtColor(rgb_prev, cv2.COLOR_GRAY2BGR)
            for (x1, y1), (_x2, _y2) in lines:
                if not x1 == _x2 and not y1 == _y2:
                    cv2.arrowedLine(vis, (x1, y1), (_x2, _y2), (0, 0, 255), 1, tipLength=0.3)
            cv2.imshow('Optical Flow Overlay', vis)

        return vis, MOVE, core_flow

    def analyse(pre_img, nxt_img):
        pyr_scale = 0.5
        levels = 3
        winsize = 15
        iterations = 3
        poly_n = 5
        poly_sigma = 1.7
        flags = 0

        pre_gray = pre_img
        nxt_gray = nxt_img
        rgb_nxt = cv2.calcOpticalFlowFarneback(
            pre_gray,
            nxt_gray,
            None,
            pyr_scale,
            levels,
            winsize,
            iterations,
            poly_n,
            poly_sigma,
            flags,
        )
        return Opticalflow.draw_flow(nxt_gray, rgb_nxt)


def measure_width(move, core_flow, rgb_prev, viz=False):
    # TODO: Horrible Results - fix
    rad = core_flow[0]
    width, height = rgb_prev.shape
    centerPt = (height / 2, width / 2)
    if not move[1] == 0 or not move[0] == 0:
        M = cv2.getRotationMatrix2D(centerPt, np.degrees(rad), 1)
        rot_img = cv2.warpAffine(rgb_prev, M, (height, width))
        indices = np.indices(rot_img.shape)
        rows = indices[0][rot_img > 0]
        cols = indices[1][rot_img > 0]
        retraction = 0
        # find the rightmost point
        if not rows.size == 0 and not cols.size == 0:
            measure_col = int(centerPt[1]) + retraction
            slice_width = np.nonzero(rot_img[:, measure_col])
            if not slice_width[0].size == 0:
                start_of_layer = slice_width[0].min()
                end_of_layer = start_of_layer
                while end_of_layer + 1 in slice_width[0]:
                    end_of_layer += 1

                width = end_of_layer - start_of_layer

                if viz:
                    width_rgb = cv2.cvtColor(rot_img, cv2.COLOR_GRAY2BGR)
                    cv2.arrowedLine(
                        width_rgb,
                        (measure_col, start_of_layer),
                        (measure_col, end_of_layer),
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow('Visualise Width Vector', width_rgb)
                return rot_img, width
    return np.zeros_like(rgb_prev), None


read_video(video_path, q3d=True, optical_flow=False)



