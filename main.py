import pickle
import os
import sys

import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.core.text import LabelBase

import cv2 as cv
import face_recognition
import numpy as np
from utils import image_resize
from identifier import get_faces_data
from collections import deque


# Declare dependencies
#Paths
sys.path.append('.')
kivy.resources.resource_add_path('.')

# Kivy Font
LabelBase.register(name = 'OpenSans', fn_regular = 'OpenSans-Regular.ttf')

# Cascades
face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
eyes_cascade = cv.CascadeClassifier('cascades/frontalEyes35x16.xml')
nose_cascade = cv.CascadeClassifier('cascades/Nose18x15.xml')

# Faces data
with open('faces-data.pickle', 'rb') as file:
    faces_data = pickle.load(file)

# Glasses + Mustache files
glasses = cv.imread("filters/glasses.png", -1)
mustache = cv.imread('filters/mustache.png',-1)

# Painter globals
# Define the upper and lower boundaries for a color to be considered "Blue"
blue_lower = np.array([100, 60, 60])
blue_upper = np.array([140, 255, 255])
# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)
# Setup deques to store separate colors in separate arrays
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
bindex = 0
gindex = 0
rindex = 0
# Drawing setting
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (0, 0, 0), (120,120,120)]
colorIndex = 0
font = cv.FONT_HERSHEY_SIMPLEX


class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv.VideoCapture(0)
        self.triger(self.update_cam)

    def triger(self, triger_mode):      #higher order function
        Clock.unschedule(self.update_cam)
        Clock.unschedule(self.detect_faces)
        Clock.unschedule(self.identify_faces)
        Clock.unschedule(self.glasses)
        Clock.unschedule(self.painter)
        Clock.schedule_interval(triger_mode, 1/60)
        
    def update_cam(self, dt):
        # repeatly update camera frame
        ret, frame = self.capture.read()
        frame = cv.resize(frame, (1090, 720), interpolation = cv.INTER_AREA)

        #convert frame to texture
        buf = cv.flip(frame, 0)
        #buf = buf.tostring()
        buf = buf.tobytes()

        texture_f = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        texture_f.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        # display frame from the texture
        self.ids.cam.texture = texture_f 

    def glasses(self, dt):
        # Apply glassess + Mustache feature

        ret, frame = self.capture.read()
        frame = cv.resize(frame, (1090, 720), interpolation = cv.INTER_AREA)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=10,
                                             minSize=(100, 100),
                                             flags=cv.CASCADE_SCALE_IMAGE)

        frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+h]
            roi_color = frame[y:y+h, x:x+h]

            eyes = eyes_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                roi_eyes = roi_gray[ey: ey + eh, ex: ex + ew]
                glasses2 = image_resize(glasses.copy(), width=ew)

                gw, gh, gc = glasses2.shape
                for i in range(0, gw):
                    for j in range(0, gh):
                        if glasses2[i, j][3] != 0: # alpha 0
                            roi_color[ey + i, ex + j] = glasses2[i, j]


            nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=20)
            for (nx, ny, nw, nh) in nose:
                roi_nose = roi_gray[ny: ny + nh, nx: nx + nw]
                mustache2 = image_resize(mustache.copy(), width=nw)

                mw, mh, mc = mustache2.shape
                for i in range(0, mw):
                    for j in range(0, mh):
                        if mustache2[i, j][3] != 0: # alpha 0
                            roi_color[ny + int(nh/2.0) + i, nx + j] = mustache2[i, j]

        # Display the resulting frame
        frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

        #convert frame to texture
        buf = cv.flip(frame, 0)
        buf = buf.tobytes()

        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        # display frame from the texture
        self.ids.cam.texture = texture1 

    def detect_faces(self, dt):
        # apply detect faces feature

        ret, frame = self.capture.read()
        frame = cv.resize(frame, (1090, 720), interpolation = cv.INTER_AREA)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=10,
                                             minSize=(100, 100),
                                             flags=cv.CASCADE_SCALE_IMAGE)

        for (x,y,w,h) in faces:
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        buf = cv.flip(frame, 0)
        buf = buf.tobytes()
        texture_f = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture_f.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.ids.cam.texture = texture_f

    def identify_faces(self, dt):
        # Apply identify faces feature

        ret, frame = self.capture.read()
        frame = cv.resize(frame, (1090, 720), interpolation = cv.INTER_AREA)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=10,
                                             minSize=(100, 100),
                                             flags=cv.CASCADE_SCALE_IMAGE)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # the facial embeddings for face in input
        encodings = face_recognition.face_encodings(rgb)
        names = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(faces_data["encodings"], encoding)
            name = "Unknown"

            if True in matches:
                #Find positions at which we get True and store them
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for each recognized face face
                for i in matched_idxs:
                    #Check the names at respective indexes we stored in matched_idxs
                    name = faces_data["names"][i]
                    #increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                #set name which has highest count
                name = max(counts, key=counts.get)
 
            # update the list of names
            names.append(name)
            # loop over the recognized faces
            for (x, y, w, h), name in zip(faces, names):
                # draw the predicted face name on the image
                cv.rectangle(frame, (x, y),(x+w, y+h),(0,255,0),2)
                cv.rectangle(frame, (x-10, y+h),(x+w+10, y+int(h*1.15)),(0,255,0), -1)
                cv.putText(frame, name, (x-5, y+int(h*1.11)), cv.FONT_HERSHEY_SIMPLEX, w/250, (255, 255, 255), 2)

        buf = cv.flip(frame, 0)
        buf = buf.tobytes()
        texture_f = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        texture_f.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.ids.cam.texture = texture_f 

    def capture_screen(self):
        # capture screen frame to gallery

        ret, frame = self.capture.read()
        DIR = './gallery'
        images = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
        count = len(images)
        while 'capture{}.jpg'.format(count) in images:
            count+=1
        cv.imwrite('gallery/capture{}.jpg'.format(count), frame)

    def painter(self, dt):
        # apply painter feature

        ret, frame = self.capture.read()
        frame = cv.resize(frame, (1090, 720), interpolation = cv.INTER_AREA)
        frame_w, frame_h, _ = frame.shape
        frame = cv.flip(frame, 1)           #Mirror
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Add the coloring options to the frame
        center_x, center_y = int(frame_w/2.5), int(frame_h/15)
        radius = 60
        space = 2*radius + 40

        #Clear button
        cv.circle(frame, (center_x, center_y), radius, colors[6], -1)
        cv.putText(frame, "CLEAR", (center_x-radius+20, center_y+10), font, .8, colors[4], 2)

        #Blue button
        cv.circle(frame, (center_x+space, center_y), radius, colors[0], -1)
        cv.putText(frame, "BLUE", (center_x+space-radius+25, center_y+10), font, .8, colors[4], 2)

        #Green button
        cv.circle(frame, (center_x+2*space, center_y), radius, colors[1], -1)
        cv.putText(frame, "GREEN", (center_x+2*space-radius+20, center_y+10), font, .8, colors[4], 2)

        #Red button
        cv.circle(frame, (center_x+3*space, center_y), radius, colors[2], -1)
        cv.putText(frame, "RED", (center_x+3*space-radius+35, center_y+10), font, .8, colors[4], 2)

        # Determine which pixels fall within the blue boundaries and then blur the binary image
        blue_mask = cv.inRange(hsv, blue_lower, blue_upper)
        blue_mask = cv.erode(blue_mask, kernel, iterations=2)
        blue_mask = cv.morphologyEx(blue_mask, cv.MORPH_OPEN, kernel)
        blue_mask = cv.dilate(blue_mask, kernel, iterations=1)

        # Find contours in the image
        cnts, _ = cv.findContours(blue_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        center = None

        # Check to see if any contours were found
        if len(cnts) > 0:
    	    # Sort the contours and find the largest one -- we
    	    # will assume this contour correspondes to the area of the bottle cap
            cnt = sorted(cnts, key = cv.contourArea, reverse = True)[0]
            # Get the radius of the enclosing circle around the found contour
            ((x, y), radius) = cv.minEnclosingCircle(cnt)
            # Draw the circle around the contour
            cv.circle(frame, (int(x), int(y)), int(radius), colors[3], 2)
            # Get the moments to calculate the center of the contour (in this case Circle)
            M = cv.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            
            global bpoints, gpoints, rpoints, bindex, gindex, rindex, colorIndex
            if center_y-radius <= center[1] <= center_y+radius:
                if center_x-radius <= center[0] <= center_x+radius: # Clear All
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]

                    bindex = 0
                    gindex = 0
                    rindex = 0

                elif center_x+space-radius <= center[0] <= center_x+space+radius:
                        colorIndex = 0 # Blue
                elif center_x+2*space-radius <= center[0] <= center_x+2*space+radius:
                        colorIndex = 1 # Green
                elif center_x+3*space-radius <= center[0] <= center_x+3*space+radius:
                        colorIndex = 2 # Red
            else :
                if colorIndex == 0:
                    bpoints[bindex].appendleft(center)
                elif colorIndex == 1:
                    gpoints[gindex].appendleft(center)
                elif colorIndex == 2:
                    rpoints[rindex].appendleft(center)

        # Append the next deque when no contours are detected (i.e., bottle cap reversed)
        else:
            bpoints.append(deque(maxlen=512))
            bindex += 1
            gpoints.append(deque(maxlen=512))
            gindex += 1
            rpoints.append(deque(maxlen=512))
            rindex += 1

        # Draw lines of all the colors (Blue, Green and Red)
        points = [bpoints, gpoints, rpoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        buf = cv.flip(frame, 0)
        buf = buf.tobytes()
        texture_f = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture_f.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.ids.cam.texture = texture_f
    
    def update_faces_data(self):
        get_faces_data("ab")

kv = Builder.load_file("screens.kv")

class FaceAPP(App):
    def build(self):
        Window.size = (1115, 540)
        Window.minimum_width = 1115
        Window.minimum_height = 540
        #Window.clearcolor = (.7, .7, .7, 1)
        #Window.borderless = "1"
        #Window.fullscreen = 'fake'
        #Window.set_system_cursor('size_we')
        Window.softinput_mode = 'resize'
        return kv

FaceAPP().run()
