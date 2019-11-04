import pickle
import cv2
import dlib
import numpy as np
import math

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

font = cv2.FONT_HERSHEY_SIMPLEX

emotions = ["neutral", "anger", "disgust", "happy", "surprise"]

video_capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")


def draw_landmarks(image):
    result = {}
    list = []
    detections = detector(image, 1)
    for k, d in enumerate(detections):
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(1, 68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=2)
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
    else:
        list.append(result['landmarks_vectorized'])
        prediction = loaded_model.predict_proba(list)
        max_val = max(prediction[0])
        if max_val > 0.5:
            cv2.putText(image, emotions[np.argmax(prediction[0])], (20, 250), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
            print("Founded emotion: ", emotions[np.argmax(prediction[0])])

    return image


while True:
    ret, frame = video_capture.read()
    image = draw_landmarks(frame)

    cv2.putText(image, "Press q to exit", (20, 20), font, 0.5, (255, 255, 255), 0, cv2.LINE_AA)
    cv2.imshow("image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break