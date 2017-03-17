from moviepy.editor import VideoFileClip

from detection import Detector
from classifiers import *
from data import *

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    X_train, X_test, scaler = normalize(X_train, X_test)

    svc = lsvc(X_train, y_train)
    booster = adaboost(X_train, y_train)
    detector = Detector((720, 1280), [svc, booster], scaler)
    output_video = 'submission.mp4'
    input_clip = VideoFileClip("project_video.mp4")
    output_clip = input_clip.fl_image(detector.process_frame)
    output_clip.write_videofile(output_video, audio=False)

