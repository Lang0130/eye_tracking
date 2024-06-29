import cv2
import dlib
import numpy as np

class FaceDetector:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect_faces(self, frame):
        dets = self.detector(frame, 1)
        for i, d in enumerate(dets):
            frame = self.show_eye(frame, d)
        return frame

    def show_eye(self, frame, d, color=(0, 255, 0)):
        cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), color, 2)
        face = frame[d.top():d.bottom(), d.left():d.right()]
        dets = self.detector(face, 3)
        for i, d in enumerate(dets):
            shape = self.predictor(face, d)

            # 目のランドマークのインデックスは36から47まで
            # 左目と右目の中心を求める
            left_eye_center = [sum(shape.part(i).x for i in range(36, 42)) // 6,
                               sum(shape.part(i).y for i in range(36, 42)) // 6]
            right_eye_center = [sum(shape.part(i).x for i in range(42, 48)) // 6,
                                sum(shape.part(i).y for i in range(42, 48)) // 6]
            # 目の中心を描画
            cv2.circle(face, tuple(left_eye_center), 2, (0, 0, 255), -1)
            cv2.circle(face, tuple(right_eye_center), 2, (0, 0, 255), -1)
            # 目の中心を描画した画像を保存
            cv2.imwrite("data/face.jpg", face)
            # 目の中心座標を表示
            print(f"Left eye center: {tuple(left_eye_center)}")
            print(f"Right eye center: {tuple(right_eye_center)}")
        return frame

    def find_black_eye_center(face, center):
        eye = face[center[1]-30:center[1]+30, center[0]-30:center[0]+30]
        gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("data/gray_eye.jpg", gray_eye)
        _, gray_eye = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY)
        cv2.imwrite("data/threshold_eye.jpg", gray_eye)
        circles = cv2.HoughCircles(gray_eye, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=10, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(face, (i[0]+center[0]-10, i[1]+center[1]-10), 2, (255, 0, 0), -1)
            cv2.imwrite("data/eye.jpg", eye)
            print(f"Black eye center: {(i[0]+center[0]-10, i[1]+center[1]-10)}")
        return frame

# # VideoCaptureオブジェクトの作成
# capture = cv2.VideoCapture(0) # 0は内蔵カメラを指定します

# # 解像度の変更
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# print(f"解像度: {capture.get(cv2.CAP_PROP_FRAME_WIDTH)}×{capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

# FaceDetectorオブジェクトの作成
predictor_path = "data/shape_predictor_68_face_landmarks.dat"
face_detector = FaceDetector(predictor_path)

#画像の読み込み
frame = cv2.imread("data/test2.jpeg")
frame = face_detector.detect_faces(frame)

#画像の保存
cv2.imwrite("data/result.jpg", frame)

# while True:
#     ret, frame = capture.read()
#     frame = face_detector.detect_faces(frame)
#     cv2.imshow('frame', frame)
#     # エスケープキーが押されたら終了
#     if cv2.waitKey(1) == 27:
#         break   

# # カメラを終了しウィンドウを閉じる
# capture.release()
# cv2.destroyAllWindows()