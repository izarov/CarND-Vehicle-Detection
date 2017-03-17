import glob
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from feature_extraction import extract_features

def read_img(fpath):
    img = mpimg.imread(fpath)
    if '.png' in fpath:
        return (img*255).astype(np.uint8)
    else:
        return img.astype(np.uint8)

def load_data():
    cars = []
    notcars = []

    images = glob.glob('data/**/*.png', recursive=True)
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)

    # Split into training and test sets
    cars_train, cars_test = train_test_split(cars)
    notcars_train, notcars_test = train_test_split(notcars)

    augmentation_factor = 0 # optionally, augment training data with randomly "jittered" examples

    X_train = []
    for img in np.concatenate((cars_train, notcars_train)):
        img = read_img(img)
        X_train.append(extract_features(img))
        for _ in range(augmentation_factor):
            X_train.append(extract_features(augmentation_pipeline(img)))

    X_train = np.array(X_train)
    y_train = np.concatenate((np.ones(len(cars_train)*(1+augmentation_factor), np.uint8), 
                              np.zeros(len(notcars_train)*(1+augmentation_factor), np.uint8)))
    X_train, y_train = shuffle(X_train, y_train)

    X_test = []
    for img in np.concatenate((cars_test, notcars_test)):
        img = read_img(img)
        X_test.append(extract_features(img))

    X_test = np.array(X_test)
    y_test = np.concatenate((np.ones_like(cars_test, np.uint8), np.zeros_like(notcars_test, np.uint8)))
    X_test, y_test = shuffle(X_test, y_test)

    return X_train, y_train, X_test, y_test

def normalize(X_train, X_test):
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler

