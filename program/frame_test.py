import cv2
import dlib
import numpy as np

class FaceDetector:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect_faces(self, frame):
        dets = self.detector(frame, 1)
        for d in dets:
            left_eye_center, right_eye_center = self.show_eye(frame, d)
            left_pupil_center, right_pupil_center = self.find_pupil(frame, d)
            left_diff = np.array(left_eye_center) - np.array(left_pupil_center)
            right_diff = np.array(right_eye_center) - np.array(right_pupil_center)
            print(f"Left eye and pupil difference: {left_diff}")
            print(f"Right eye and pupil difference: {right_diff}")
            print("\n")

        return frame

    def show_eye(self, frame, d, color=(0, 255, 0)):
        shape = self.predictor(frame, d)
        left_eye_center = self.calculate_eye_center(shape, range(36, 42))
        right_eye_center = self.calculate_eye_center(shape, range(42, 48))
        print(f"Left eye center: {tuple(left_eye_center)}")
        print(f"Right eye center: {tuple(right_eye_center)}")
        return left_eye_center, right_eye_center

    def calculate_eye_center(self, shape, eye_landmarks):
        eye_landmarks_x = [shape.part(i).x for i in eye_landmarks]
        eye_landmarks_y = [shape.part(i).y for i in eye_landmarks]
        eye_center = (sum(eye_landmarks_x) // len(eye_landmarks), sum(eye_landmarks_y) // len(eye_landmarks))
        return eye_center

    def find_pupil(self, frame, d, color=(0, 0, 255)):
        shape = self.predictor(frame, d)
        left_eye = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
        right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
        left_eye_img = self.extract_eye(frame, left_eye)
        right_eye_img = self.extract_eye(frame, right_eye)
        left_pupil_center = self.find_pupil_center(left_eye_img)
        right_pupil_center = self.find_pupil_center(right_eye_img)
        cv2.circle(frame, left_pupil_center, 1, color, -1)
        cv2.circle(frame, right_pupil_center, 1, color, -1)
        print(f"Left pupil center: {left_pupil_center}")
        print(f"Right pupil center: {right_pupil_center}")

        if left_pupil_center[0] > 0 and right_pupil_center[0] > 0:
            direction = "左"
        elif left_pupil_center[0] < 0 and right_pupil_center[0] < 0:
            direction = "右"
        else:
            direction = "正面"

        if left_pupil_center[1] > 0 and right_pupil_center[1] > 0:
            direction += "上"
        elif left_pupil_center[1] < 0 and right_pupil_center[1] < 0:
            direction += "下"
        else:
            direction += "正面"
        print(f"Direction: {direction}")
        return left_pupil_center, right_pupil_center

    def extract_eye(self, frame, eye_landmarks):
        eye_region = np.array(eye_landmarks, np.int32)
        eye_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(eye_mask, eye_region, 255)
        eye_img = cv2.bitwise_and(frame, frame, mask=eye_mask)
        return eye_img

    def find_pupil_center(self, eye_img):
        center = (0, 0)
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_not(binary)
        y, x = np.where(binary == 0)
        if len(x) > 0 and len(y) > 0:
            center = (int(np.mean(x)), int(np.mean(y)))
        return center
    
def main():
    face_detector = FaceDetector("data/shape_predictor_68_face_landmarks.dat")
    image = cv2.imread("data/png/up.png")
    if image is None:
        print("Failed to load image")
        exit()
    frame = cv2.resize(image, (1280, 720))
    frame = face_detector.detect_faces(frame)
    cv2.imshow("Frame", frame)

if __name__ == "__main__":
    main()