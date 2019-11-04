# showing how the facial landmarks detector works

import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")


def draw_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image, 1)
    for k, d in enumerate(detections):
        shape = predictor(clahe_image, d)
        for i in range(1, 68):
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=2)
    return image


font = cv2.FONT_HERSHEY_SIMPLEX
video_capture = cv2.VideoCapture(0)

camera_on = True

video_capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")

while True:

    if camera_on:
        if not video_capture.isOpened():
            video_capture = cv2.cv2.VideoCapture(0)

        ret, frame = video_capture.read()
        image = draw_landmarks(frame)
        cv2.putText(image, "Press i to test static image", (20, 80), font, 0.55, (255, 255, 255), 0, cv2.LINE_AA)
    else:
        static_image = cv2.imread("assets/test/images/1.jpg")
        image = draw_landmarks(static_image)
        video_capture.release()
        cv2.putText(image, "Press c to text camera", (20, 80), font, 0.55, (255, 255, 255), 0, cv2.LINE_AA)

    cv2.putText(image, "Press q to exit", (20, 50), font, 1, (255, 255, 255), 0, cv2.LINE_AA)
    cv2.imshow("image", image)

    key = cv2.waitKey(1)
    if key == ord('c'):
        camera_on = True
    elif key == ord('i'):
        camera_on = False
    elif key == ord('q'):
        break

