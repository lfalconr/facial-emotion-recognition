import cv2
import glob
import random
import math
import numpy as np
import dlib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import pickle

#               0          1        2           3         4        5         6          7
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

#                      0          1         2         3         4
reduced_emotions = ["neutral", "anger", "disgust", "happy", "surprise"]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")

clf_linear = SVC(kernel="linear", probability=True, tol=1e-3)
clf_polynomial = SVC(kernel="poly", probability=True, tol=1e-3, degree=2)
clf_rbf = SVC(kernel="rbf", probability=True, tol=1e-3)
clf_sigmoid = SVC(kernel="sigmoid", probability=True, tol=1e-3)

data = {}


def draw_data_graph(emotions):
    y_pos = np.arange(len(emotions))
    count = [len(glob.glob("assets\\sorted_set\\%s\\*.png" % emotion)) for emotion in emotions]

    plt.bar(y_pos, count, align='center', alpha=0.5)
    plt.xticks(y_pos, emotions)
    plt.ylabel('Count')
    plt.title('Data distribution')
    plt.show()


draw_data_graph(emotions)

def get_files(emotion):
    files = glob.glob("assets\\sorted_set\\%s\\*.png" % emotion)
    random.shuffle(files)
    print("  files count %:", len(files))
    training = files[:int(len(files) * 0.8)]
    prediction = files[-int(len(files) * 0.2):]
    return training, prediction


def get_landmarks(image):
    result = {}
    detections = detector(image, 1)
    for k, d in enumerate(detections):
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(1, 68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray(ymean, xmean)
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))
        result['landmarks_vectorized'] = landmarks_vectorised
    if len(detections) < 1:
        result['landmarks_vectorized'] = "error"
    return result


def make_set(emotions):
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    result = {}

    for emotion in emotions:
        print(" working on %s" % emotion)
        training, prediction = get_files(emotion)

        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            clahe_image = clahe.apply(gray)
            result = get_landmarks(clahe_image)
            if result['landmarks_vectorized'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(result["landmarks_vectorized"])
                training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            result = get_landmarks(clahe_image)
            if result['landmarks_vectorized'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(result["landmarks_vectorized"])
                prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels


# train the models
def train_model(emotions):
    accur_lin = []
    accur_polinomial = []
    accur_rbf = []
    accur_sigmoid = []

    for i in range(0, 1):  # 10
        print("Making sets %s" % i)
        training_data, training_labels, validation_data, validation_labels = make_set(emotions)
        npar_train = np.array(training_data)
        npar_trainlabs = np.array(training_labels)
        print("training SVM linear %s" % i)  # train SVM
        clf_linear.fit(npar_train, npar_trainlabs)
        print("training SVM polinomial %s" % i)  # train SVM
        clf_polynomial.fit(npar_train, npar_trainlabs)
        print("training SVM rbf %s" % i)  # train SVM
        clf_rbf.fit(npar_train, npar_trainlabs)
        print("training SVM sigmoid %s" % i)  # train SVM
        clf_sigmoid.fit(npar_train, npar_trainlabs)
        print("getting accuracies %s" % i)
        npar_pred = np.array(validation_data)
        pred_lin = clf_linear.score(npar_pred, validation_labels)
        pred_polynomial = clf_polynomial.score(npar_pred, validation_labels)
        pred_rbf = clf_rbf.score(npar_pred, validation_labels)
        pred_sigmoid = clf_sigmoid.score(npar_pred, validation_labels)
        print("linear: ", pred_lin)
        print("polynomial: ", pred_polynomial)
        print("rbf: ", pred_rbf)
        print("sigmoid: ", pred_rbf)
        accur_lin.append(pred_lin)
        accur_polinomial.append(pred_polynomial)
        accur_rbf.append(pred_rbf)
        accur_sigmoid.append(pred_sigmoid)

    mean_linear = np.mean(accur_lin)
    mean_polynomial = np.mean(accur_polinomial)
    mean_rbf = np.mean(accur_rbf)
    mean_sigmoid = np.mean(accur_sigmoid)
    print("Mean value linear svm: %s" % mean_linear)
    print("Mean value polynomial svm: %s" % mean_polynomial)
    print("Mean value rbf svm: %s" % mean_rbf)
    print("Mean value sigmoid svm: %s" % mean_sigmoid)

    draw_comparative_graph(mean_linear, mean_polynomial, mean_rbf, mean_sigmoid)
    draw_confusion_matrix(emotions, validation_data, validation_labels)


def draw_comparative_graph(mean_linear, mean_polynomial, mean_rbf, mean_sigmoid):
    kernels = ('Linear', 'Polynomial', 'Rbf', 'Sigmoid')
    y_pos = np.arange(len(kernels))
    acc = [mean_linear, mean_polynomial, mean_rbf, mean_sigmoid]

    plt.bar(y_pos, acc, align='center', alpha=0.5)
    plt.xticks(y_pos, kernels)
    plt.ylabel('Accuracy')
    plt.title('SVM Kernels results')
    plt.show()


def draw_confusion_matrix(emotions, validation_data, validation_labels):
    prediction_validation = clf_linear.predict(validation_data)
    conf_matrix = confusion_matrix(prediction_validation, validation_labels)
    print(conf_matrix)
    print(classification_report(prediction_validation, validation_labels))

    cm_df = pd.DataFrame(conf_matrix, emotions, emotions)
    plt.figure(figsize=(20, 16))
    sn.heatmap(cm_df, annot=True)

    # df_cm = pd.DataFrame(conf_matrix, range(len(emotions)), range(len(emotions)))
    # # plt.figure(figsize = (15,10))
    # sn.set(font_scale=1.4)  # for label size
    # sn.heatmap(df_cm, annot = True, annot_kws=[emotions])  # font size
    plt.show()

print("Complete array of emotions")
train_model(emotions)
print("Reduced array of emotions")
train_model(reduced_emotions)

def prepare_images(path):
    files = glob.glob(path)
    images_landmarks = []
    images_names = []
    for file in files:
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        clahe_image = clahe.apply(gray)
        result = get_landmarks(clahe_image)

        if result['landmarks_vectorized'] == "error":
            print("no face detected on this one %s" % file)
        else:
            images_names.append(file)
            images_landmarks.append(result['landmarks_vectorized'])
    return images_names, images_landmarks


def print_result(emotions, names, results):
    for name, result in zip(names, results):
        print("Emotion in {} is {}".format(name, emotions[result]))


def print_prob_result(emotions, names, results):
    for name, result in zip(names, results):
        print("Image {}".format(name))
        for emotion, percentage in zip(emotions, result):
            print("         emotion {} probability {}".format(emotion, percentage))


# test ekman images
ekman_images, ekman_landmarks = prepare_images("assets//test//ekman//*")
results_ekman = clf_linear.predict(ekman_landmarks)
print_result(reduced_emotions, ekman_images, results_ekman)

print("----------------------------------")

# test google images
test_images, test_landmarks = prepare_images("assets//test//images//*")
results_test = clf_linear.predict(test_landmarks)
results_prob_test = clf_linear.predict_proba(test_landmarks)
print_result(reduced_emotions, test_images, results_test)
print_prob_result(reduced_emotions, test_images, results_prob_test)


# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf_linear, open(filename, 'wb'))
