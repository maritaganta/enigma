import cv2
import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN



video_path = 'data/animation_2.mp4'

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

            try:
                prev_frame = current_frame
            except:
                pass
            current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.circle(frame, (frame.shape[1]//2, frame.shape[0]//2), 1, (0,0,255), 2)
            cv2.imshow('Playback Video', frame)

            if q3d:

                frame = preprocess(frame)
                frame = crop_square(frame)
                cnts = contouring(frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                fit_curve(frame, cnts)
                #cv2.drawContours(frame, cnts, -1, (0, 255, 0), 3)
                cv2.imshow('Contour Overlay', frame)

            if optical_flow:
                try:
                    flow_img, move, core_flow = Opticalflow.analyse(prev_frame, current_frame)
                except:
                    pass

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def preprocess(frame):
    _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    frame = cv2.bitwise_not(frame)
    return frame



def crop_square(frame, size=(480,480)):

    w, h = frame.shape
    center = (h//2, w//2)

    top_left_x = center[0] - size[0] // 2
    top_left_y = center[1] - size[1] // 2

    bottom_right_x = top_left_x + size[0]
    bottom_right_y = top_left_y + size[1]

    cropped_frame = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    return cropped_frame


def contouring(frame):
    contours, _ = cv2.findContours(frame, 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_NONE) 
    
    return contours


def objective(x, a, b, c, d):
    return a * np.sin(b - x) + c * x**2 + d

def fit_curve(frame, cnts):

    if len(cnts)>0:
        cnt = cnts[0]

        x_values =[]
        y_values = []
        for point in cnt:
            x = point[0][0]
            y = point[0][1]
            if x != 0 and y !=0 and x!=479 and y!=479:
                x_values.append(point[0][0])
                y_values.append(point[0][1])
                cv2.circle(frame, (x,y), 1, (0,0,255), 2)

        popt, _ = curve_fit(objective, x_values, y_values)

        a, b, c, d = popt

        x_line = np.arange(0, frame.shape[0], 1)
        y_line = objective(x_line, a, b, c, d)

        # TODO: Ensure the y_line is being parsed properly in higher order polynomial (values within image bounds)
        for i, element in enumerate(x_line):
            cv2.circle(frame, (int(x_line[i]), int(y_line[i])),1,(255,0,255), 1)


class Opticalflow:
    def draw_flow(rgb_prev, rgb_nxt):
        THRESHOLD_DETECT = 20
        THRESHOLD_IGNORE_L = 1
        THRESHOLD_IGNORE_H = 300
        MOVE = [0, 0]
        COUNT_MOVE = 1
        STEP = 20
        # generate grids
        h, w = rgb_prev.shape[:2]
        y, x = np.mgrid[STEP / 2 : h : STEP, STEP / 2 : w : STEP].reshape(2, -1).astype(int)
        fx, fy = rgb_nxt[y, x].T

        angle = np.arctan2(fy, fx)

        # clustering
        cluster_lables = DBSCAN(eps=0.01, min_samples=2).fit(angle.reshape(-1, 1))
        core_cluster = angle[cluster_lables != -1]
        core_flow = np.mean(core_cluster, axis=1)

        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(rgb_prev, cv2.COLOR_GRAY2BGR)

        for (x1, y1), (_x2, _y2) in lines:
            if not x1 == _x2 and not y1 == _y2:
                cv2.arrowedLine(vis, (x1, y1), (_x2, _y2), (0, 0, 255), 1, tipLength=0.3)

        # calculate move
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

        cv2.imshow('Optical flow', vis)
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
    
    @staticmethod
    def measure_width(move, angle, depth_img):
        rad = angle
        width, height = depth_img.shape
        centerPt = (height / 2, width / 2)
        if not move[1] == 0 or not move[0] == 0:
            M = cv2.getRotationMatrix2D(centerPt, degrees(rad), 1)
            rot_img = cv2.warpAffine(depth_img, M, (height, width))
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

                    depth_rgb = cv2.cvtColor(rot_img, cv2.COLOR_GRAY2BGR)
                    cv2.arrowedLine(
                        depth_rgb,
                        (measure_col, start_of_layer),
                        (measure_col, end_of_layer),
                        (0, 0, 255),
                        2,
                    )
                    return depth_rgb, width
        return np.zeros_like(depth_img), None


read_video(video_path, q3d=False, optical_flow=True)



