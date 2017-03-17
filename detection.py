import numpy as np
import cv2
from scipy.ndimage.measurements import label

from heatmap import RunningHeatmap
from windows import *
from centroid import Centroid

class Detector:
    """
    Detector is the entry class for car detection.
    
    It keeps a running heatmap of detections in the image, 
    as well as a set of centroids for all detected objects.
    """
    MAX_INACTIVITY = 60 # max number of frames without activity
    MIN_ACTIVITY_COUNT = 3 # min number of activations before drawing
    
    def __init__(self, shape, classifiers, scaler):
        self.heatmap = RunningHeatmap(shape)
        self.classifiers = classifiers
        self.scaler = scaler
        self.centroids = set()
        
    def process_frame(self, image):
        """
        Given the next frame in a video returns the same image with
        boxes drawn around all detected cars.
        """
        # get sliding windows
        windows = slide_window(image, x_start_stop=[400, None], xy_window=(112, 112), xy_overlap=(0.6, 0.6))
        # search all windows for cars and get the 'hot' windows with a positive detection
        hot_windows = search_windows(image, windows, self.classifiers, self.scaler)
        # update the running heatmap with the latest hot windows
        self.heatmap.add(hot_windows)
        
        # get a labeled array from the heatmap, merging neighboring pixels which are
        # considered to overlap as part of a single object with multiple hot windows
        labels = label(self.heatmap.heatmap)
        boxes = []
        centroid_pts = []
        for i in range(1, labels[1]+1):
            nonzeroy, nonzerox = (labels[0] == i).nonzero()
            box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            centroid_pts.append(((box[0][0]+box[1][0])//2, (box[0][1]+box[1][1])//2))
            boxes.append(box)
        
        found_centroids = set()
        
        # loop through all detected boxes and their centroid points
        # and allocate points to current centroid objects if they fall in range
        for i, cpt in enumerate(centroid_pts):
            found = False
            for centroid in self.centroids:
                if centroid.near(cpt):
                    centroid.set_active()
                    centroid.update_box(boxes[i])
                    found = True
                    found_centroids.add(centroid)
            
            # if no capturing centroid object found, create a new one
            if not found:
                new_centroid = Centroid(boxes[i])
                self.centroids.add(new_centroid)
                found_centroids.add(new_centroid)
                
        # clean up old detections
        dead_centroids = set()
        for centroid in self.centroids - found_centroids:
            centroid.set_inactive()
            if len(centroid.activations) > self.MAX_INACTIVITY and not any(centroid.activations[-self.MAX_INACTIVITY:]):
                dead_centroids.add(centroid)
        
        for c in dead_centroids:
            self.centroids.remove(c)
        
        # draw boxes for objects with the minimum number of positive detections
        for centroid in self.centroids:
            if np.sum(centroid.activations) >= self.MIN_ACTIVITY_COUNT:
                cv2.rectangle(image, centroid.draw_box[0], centroid.draw_box[1], (0,0,255), 6)
        
        return image
