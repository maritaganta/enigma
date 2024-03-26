import cv2
import numpy as np
from scipy.optimize import curve_fit


video_path = 'data/animation_2.mp4'

def read_video(video_path, q3d=False):
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Could not open video")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:

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


            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

        print(y_line.shape, type(x_line[0]))

        for i, element in enumerate(x_line):
            cv2.circle(frame, (int(x_line[i]), int(y_line[i])),1,(255,0,255), 1)


read_video(video_path, q3d=True)



