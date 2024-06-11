import numpy as np
import cv2
from main import features_fit_curve, generate_x_line, line_calculation


video_path = 'data/real_data.webm'

CENTER = (343, 343)  # TODO: Proper benchmark

def read_video(video_path):
    
    cap = cv2.VideoCapture(video_path)

    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]

    if not cap.isOpened():
        print("ERROR: Could not open video")
        return
    
    while cap.isOpened():

        ret, frame = cap.read()

        cv2.imshow('Playback Video', frame)

        if not ret:
            break

        frame = testing(frame, CENTER)

        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def testing(frame, center):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

    frame = padding(frame, center)

    new = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)

    copy = isolate_area(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
    copy = cv2.morphologyEx(copy, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(copy, 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE) 
    
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    
    counter = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            counter += 1

    if counter > 1:
        print("Generalisation Eror")
        # TODO: Boolean that discards timestamp
    else:                
        epsilon = 0.02*cv2.arcLength(contours[-1],True)
        approx = cv2.approxPolyDP(contours[-1],epsilon,True)

        cv2.drawContours(new,[approx],-1,(0,255,0),2)

        M = cv2.moments(approx)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        theta = 0.5*np.arctan2(2*M["mu11"],M["mu20"]-M["mu02"])
        endx = 600 * np.cos(theta) + center[0] 
        endy = 600 * np.sin(theta) + center[1]

        start_point = center
        end_point = (int(endx),int(endy))

        length = 50

        cv2.line(new,start_point,end_point,(255,0,0),2)

        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]

        # Calculate the perpendicular direction vector
        perp_dx = -dy
        perp_dy = dx

        # Normalize the perpendicular direction vector
        magnitude = np.sqrt(perp_dx**2 + perp_dy**2)
        perp_dx /= magnitude
        perp_dy /= magnitude

        # Calculate both end points of the perpendicular line
        perp_end_point1 = (int(start_point[0] + perp_dx * length), int(start_point[1] + perp_dy * length))
        perp_end_point2 = (int(start_point[0] - perp_dx * length), int(start_point[1] - perp_dy * length))

        # Draw the perpendicular line
        cv2.line(new, perp_end_point1, perp_end_point2, (0, 255, 0), 2)

    cv2.imshow('Processed', new)


    return frame

def padding(frame, center):

    original_height, original_width = frame.shape

    if center[1] - original_height // 2 > 0:
        vertical = center[1]
    else:
        vertical = original_height - center[1]

    if center[0] - original_width // 2 > 0:
        horizontal = center[0]
    else:
        horizontal = original_width - center[0]

    new_image = np.zeros((2*vertical, 2*horizontal), dtype=np.uint8)

    new_image[0:frame.shape[0], 0:frame.shape[1]] = frame

    return new_image


def isolate_area(frame):

    height, width = frame.shape

    # Define the size of the rectangle
    rect_height = 300
    rect_width = 300

    # Calculate the center of the original image
    center_y, center_x = height // 2, width // 2

    # Calculate the top-left corner of the rectangle
    top_left_y = max(0, center_y - rect_height // 2)
    top_left_x = max(0, center_x - rect_width // 2)

    # Calculate the bottom-right corner of the rectangle
    bottom_right_y = min(height, top_left_y + rect_height)
    bottom_right_x = min(width, top_left_x + rect_width)

    copy = np.zeros((frame.shape), dtype=np.uint8)        

    copy[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    return copy



read_video(video_path)

