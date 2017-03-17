import os.path
import pickle
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

def lsvc(X_train, y_train):
    if os.path.exists('lsvcclassifier.p'):
        lsvc = pickle.load(open('lsvcclassifier.p', 'rb'))
    else:
        lsvc = LinearSVC(C=0.25)
        lsvc.fit(X_train, y_train)
        pickle.dump(lsvc, open('lsvcclassifier.p', 'wb'))

    return lsvc

def adaboost(X_train, y_train):
    if os.path.exists('adaboostclassifier.p'):
        booster = pickle.load(open('adaboostclassifier.p', 'rb'))
    else:
        booster = AdaBoostClassifier()
        booster.fit(X_train, y_train)
        pickle.dump(booster, open('adaboostclassifier.p', 'wb'))

    return booster
