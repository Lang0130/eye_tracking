import cv2
import dlib
import numpy as np

class FaceDetector:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)


    # 画像比率を維持しながら縦横を指定して拡大する 余った部分は白でパディング 
    def resize_with_pad(image,new_shape,padding_color=[255,255,255]):
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(new_shape))/max(original_shape)
        new_size = tuple([int(x*ratio) for x in original_shape])
        image = cv2.resize(image, new_size)
        delta_w = new_shape[0] - new_size[0]
        delta_h = new_shape[1] - new_size[1]
        top, bottom = delta_h//2 + delta_h-(delta_h//2), 0
        left, right = delta_w//2, delta_w-(delta_w//2)
        top, bottom, left, right = max(top,0),max(bottom,0),max(right,0),max(left,0)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
        return image

    def detect_faces(self, frame):
        try:
            dets = self.detector(frame, 1)
            # # ランドマークを描画
            # for d in dets:
            #     shape = self.predictor(frame, d)
            #     for i in range(68):
            #         # 番号を描画
            #         cv2.putText(frame, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            # cv2.imwrite("detected_face.png", frame)
            if len(dets) == 0:
                print("No faces detected")
                return frame
            




            return frame
        except Exception as e:
            print(f"Error in detect_faces: {e}")
            return frame

    def slice_eye(self, frame, d, color=(0, 255, 0)):
        try:
            shape = self.predictor(frame, d)
            left_eye = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
            right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

            left_eye_img = self.extract_eye(frame, left_eye)
            right_eye_img = self.extract_eye(frame, right_eye)

            # 検出した目以外の部分を白く塗りつぶす
            mask = np.zeros(left_eye_img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(left_eye, np.int32)], 255)
            cv2.fillPoly(mask, [np.array(right_eye, np.int32)], 255)
            mask = cv2.bitwise_not(mask)
            frame[mask == 255] = (255, 255, 255)
            detected_eye = frame.copy()

            cv2.imwrite("detected_eye.png", frame)

            # 目を切り取った画像を保存
            cv2.imwrite("left_eye.png", left_eye_img)
            cv2.imwrite("right_eye.png", right_eye_img)

            return left_eye_img, right_eye_img, detected_eye
        except Exception as e:
            print(f"Error in slice_eye: {e}")
            return None, None

    
    def extract_eye(self, frame, eye_landmarks):
        try:
            eye_region = np.array(eye_landmarks, np.int32)
            eye_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(eye_mask, eye_region, 255)
            eye_img = cv2.bitwise_and(frame, frame, mask=eye_mask)
            return eye_img
        except Exception as e:
            print(f"Error in extract_eye: {e}")
            return None
    
    def find_eye_center(self, eye_img):
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

        # 白黒反転
        binary = cv2.bitwise_not(binary)

        # 画像の中から円を検出
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours) == 0:
            print("No circles detected")
            return None
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if 10 < radius < 30:
                break
        # 検出した円を描画
        cv2.circle(eye_img, center, radius, (0, 255, 0), 2)

        # 円の中心を描画
        cv2.circle(eye_img, center, 1, (0, 255, 0), -1)

        print(f"Eye center: {center}")
        cv2.imwrite(f"{eye_img}_find_circle.png", eye_img)
        return center
    


def main():
    face_detector = FaceDetector("data/shape_predictor_68_face_landmarks.dat")
    image = cv2.imread("data/img/image.png")
    if image is None:
        print("Failed to load image")
        return
    frame = face_detector.detect_faces(image)
    # cv2.imshow("Frame", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
