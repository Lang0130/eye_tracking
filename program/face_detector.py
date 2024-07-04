import cv2
import dlib
import numpy as np

class FaceDetector:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()  # 顔検出器を初期化
        self.predictor = dlib.shape_predictor(predictor_path)  # ランドマーク予測器を初期化

    def detect_faces(self, frame):
        dets = self.detector(frame, 1)  # 顔検出を実行
        for d in dets:
            left_eye_center, right_eye_center = self.show_eye(frame, d)
            left_pupil_center, right_pupil_center = self.find_pupil(frame, d)
            # 各座標のずれを計算
            left_diff = np.array(left_eye_center) - np.array(left_pupil_center)
            right_diff = np.array(right_eye_center) - np.array(right_pupil_center)
            print(f"Left eye and pupil difference: {left_diff}")
            print(f"Right eye and pupil difference: {right_diff}")
            print("\n")

            if left_diff[0] > 0:
                print("Left: Right") 
        return frame

    def show_eye(self, frame, d, color=(0, 255, 0)):
        shape = self.predictor(frame, d)  # 顔のランドマークを予測

        # 左目と右目の中心を計算
        left_eye_center = [sum([shape.part(i).x for i in range(36, 42)]) // 6,
                           sum([shape.part(i).y for i in range(36, 42)]) // 6]
        right_eye_center = [sum([shape.part(i).x for i in range(42, 48)]) // 6,
                            sum([shape.part(i).y for i in range(42, 48)]) // 6]
        
        # 目の中心座標を表示
        print(f"Left eye center: {tuple(left_eye_center)}")
        print(f"Right eye center: {tuple(right_eye_center)}")

        return left_eye_center, right_eye_center

    def find_pupil(self, frame, d, color=(0, 0, 255)):
        shape = self.predictor(frame, d)  # 顔のランドマークを予測

        # 左目と右目のランドマーク座標を取得
        left_eye = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
        right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

        # 左目を切り出す
        left_eye_region = np.array(left_eye, np.int32)
        left_eye_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(left_eye_mask, left_eye_region, 255)
        left_eye_img = cv2.bitwise_and(frame, frame, mask=left_eye_mask)

        # 右目を切り出す
        right_eye_region = np.array(right_eye, np.int32)
        right_eye_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(right_eye_mask, right_eye_region, 255)
        right_eye_img = cv2.bitwise_and(frame, frame, mask=right_eye_mask)

        # 瞳孔の中心を計算
        left_pupil_center = self.find_pupil_center(left_eye_img, "left")
        right_pupil_center = self.find_pupil_center(right_eye_img, "right")

        # # 瞳孔の中心を描画
        cv2.circle(frame, left_pupil_center, 10, color, -1)
        cv2.circle(frame, right_pupil_center, 10, color, -1)

        # 瞳孔の中心座標を表示
        print(f"Left pupil center: {left_pupil_center}")
        print(f"Right pupil center: {right_pupil_center}")

        return left_pupil_center, right_pupil_center

    def find_pupil_center(self, eye_img, eye_side):
        center = (0, 0)
        # グレースケールに変換
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)

        # 2値化
        thre = 60
        _, binary = cv2.threshold(gray, thre, 255, cv2.THRESH_BINARY)

        # 白黒反転
        binary = cv2.bitwise_not(binary)

        # 黒い領域の平均座標を計算
        y, x = np.where(binary == 0)
        if len(x) > 0 and len(y) > 0:
            center = (int(np.mean(x)), int(np.mean(y)))
        else:
            pass

        return center
    
if __name__ == "__main__":
    # 顔検出器の初期化
    face_detector = FaceDetector("data/shape_predictor_68_face_landmarks.dat")

    # macの内蔵カメラを使用してVideoCaptureオブジェクトの作成
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("カメラが見つかりません")
        exit()

    # 解像度の変更
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    print(f"解像度: {capture.get(cv2.CAP_PROP_FRAME_WIDTH)}×{capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    count_sec = 0
    while True:
        # カメラから1フレーム読み込む
        ret, frame = capture.read()
        if count_sec % 30 == 0:
            frame = face_detector.detect_faces(frame)
        cv2.imshow('frame', frame)
        count_sec += 1

        # エスケープキーが押されたら終了
        if cv2.waitKey(1) == 27:
            break


    # カメラを終了しウィンドウを閉じる
    capture.release()
    cv2.destroyAllWindows()