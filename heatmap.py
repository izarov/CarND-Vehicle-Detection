import numpy as np

class RunningHeatmap:
    """
    Container for a heatmap formed by a sequence of box predictions.
    """
    def __init__(self, shape, history=60, threshold=3):
        self.history = history
        self.threshold = threshold
        self._heatmap = np.zeros(shape, np.uint32)
        self._boxes = []
        
    def add(self, boxes):
        """
        Add a list of boxes to the running heatmap.
        
        Method automatically removes old boxes if max history exceeded.
        """
        self._boxes.append(boxes)
        if len(self._boxes) > self.history:
            popped = self._boxes.pop(0)
            for box in popped:
                self._heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] -= 1
        
        for box in boxes:
            self._heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
            
    @property
    def heatmap(self):
        """
        Get the current tresholded value of the heatmap.
        """
        threshold = max(min(self.threshold, len(self._boxes)//2), 3)
        ret = np.clip(self._heatmap, 0, 255).astype(np.uint8)
        ret[ret <= self.threshold] = 0
        return ret
