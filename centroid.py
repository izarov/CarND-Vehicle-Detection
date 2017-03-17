class Centroid:
    """
    Container class which tracks an object via its centroid.
    
    New detections which fall within the range of this object are
    considered to be a detection of the same object.
    """
    MAX_PIXEL_DISTANCE = 100
    
    def __init__(self, box):
        self.previous_box = box
        self.box = box
        self.draw_box = box
        self.activations = [True]
        self.update_center()
        
    def update_center(self):
        """
        Recaculate the center of the centroid after receiving a new box.
        """
        self.center = ((self.box[0][0]+self.box[1][0])//2, (self.box[0][1]+self.box[1][1])//2)

    def near(self, point):
        """
        Check if a point falls in the range of this centroid.
        
        Uses Euclidean distance.
        """
        distance = ((self.center[0]-point[0])**2 + (self.center[1]-point[1])**2)**0.5
        return distance <= self.MAX_PIXEL_DISTANCE
    
    def update_box(self, new_box):
        """
        Update the bounding box of the object with a new observation.
        """
        # take the average of the current box position with it's new position
        self.draw_box = (((self.box[0][0]+new_box[0][0])//2, (self.box[0][1]+new_box[0][1])//2),
                        ((self.box[1][0]+new_box[1][0])//2, (self.box[1][1]+new_box[1][1])//2))
        self.previous_box = self.box
        self.box = new_box
        self.update_center()
        
    def set_active(self):
        """
        Mark centroid as active at this time step.
        """
        self.activations.append(True)
        
    def set_inactive(self):
        """
        Mark centroid as inactive (not detected) at this time step.
        """
        self.activations.append(False)
        
