import dlib
import cv2
import numpy as np

thresh = 50

detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "data/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

cap = cv2.VideoCapture(0)
print(cap.get(cv2.CAP_PROP_FPS))

x_max = None
x_min = None
y_max = None
y_min = None

x_r = None
x_l = None
y_t = None
y_b = None

capture = cv2.VideoCapture(0)

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
    return image

# 左目 36:左端 37, 38:上 39:右端 40,41:下
# 右目 42:左端 43, 44:上 45:右端 46,47:下
def show_eye(img, parts,xy=[None,None,None,None]):
    # 左目の左上のカット
    delta = (parts[37].y-parts[36].y)/(parts[37].x-parts[36].x)
    y = parts[36].y
    for x in range(parts[36].x,parts[37].x+1):
        img[0:round(y),x] = 255
        y += delta
        
    # 左目の左下のカット
    delta = (parts[41].y-parts[36].y)/(parts[41].x-parts[36].x)
    y = parts[36].y
    for x in range(parts[36].x,parts[41].x+1):
        img[round(y):,x] = 255
        y += delta
        
    # 左目の上部のカット
    delta = (parts[38].y-parts[37].y)/(parts[38].x-parts[37].x)
    y = parts[37].y
    for x in range(parts[37].x,parts[38].x+1):
        # print(x,round(y))
        img[0:round(y),x] = 255
        y += delta
        
    # 左目の右上部のカット
    delta = (parts[39].y-parts[38].y)/(parts[39].x-parts[38].x)
    y = parts[38].y
    for x in range(parts[38].x,parts[39].x+1):
        # print(x,round(y))
        img[0:round(y),x] = 255
        y += delta
        
    # 左目の右下部のカット
    delta = (parts[39].y-parts[40].y)/(parts[39].x-parts[40].x)
    y = parts[40].y
    for x in range(parts[40].x,parts[39].x+1):
        # print(x,round(y))
        img[round(y):,x] = 255
        y += delta
        
    # 左目の下部のカット
    delta = (parts[41].y-parts[40].y)/(parts[41].x-parts[40].x)
    y = parts[41].y
    for x in range(parts[41].x,parts[40].x+1):
        # print(x,round(y))
        img[round(y):,x] = 255
        y += delta
    
    # 目の位置を求める
    x0_right = parts[36].x
    x1_right = parts[39].x
    y0_right = min(parts[37].y, parts[38].y)
    y1_right = max(parts[40].y, parts[41].y)

    x0_left = parts[42].x
    x1_left = parts[45].x
    y0_left = min(parts[43].y, parts[44].y)
    y1_left = max(parts[46].y, parts[47].y)
    
    # 目の長方形をスライスで切り出す
    right_eye = img[y0_right:y1_right, x0_right:x1_right]
    left_eye = img[y0_left:y1_left, x0_left:x1_left]
    
    # そのままの大きさでは見づらいので拡大する
    right_eye = resize_with_pad(right_eye,(600,300),padding_color=[255,255,255])
    left_eye = resize_with_pad(left_eye,(600,300),padding_color=[255,255,255])
    
    # 右目の重心を求めるために二値化する
    img_gray = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)

    # 重心を求める
    img_rev = 255 - img_bin # 白黒を反転する
    mu = cv2.moments(img_rev, False) # 反転した画像の白い部分の重心を求める
    
    x_right,y_right = None,None
    
    try:
        x_right,y_right= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
    except:
        pass

    # 左目の重心を求めるために二値化する
    img_gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    # ret2, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

    # 重心を求める
    img_rev = 255 - img_bin # 白黒を反転する
    mu = cv2.moments(img_rev, False) # 反転した画像の白い部分の重心を求める

    x_left,y_left = None,None

    try:
        x_left,y_left= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
    except:
        pass

    return x_right,y_right,x_left,y_left

def set_xy(x_max,x_min,y_max,y_min):
    if None not in [x_max,x_min,y_max,y_min]:
        x_r = x_min + (x_max-x_min)/3
        x_l = x_min + 2*(x_max-x_min)/3
        y_t = y_min + (y_max-y_min)/3
        y_b = y_min + 2*(y_max-y_min)/3
        print(x_r,x_l,y_t,y_b)
        return round(x_r),round(x_l),round(y_t),round(y_b)
    else:
        return None,None,None,None

if not capture.isOpened():
    print("カメラが見つかりません")
    exit()
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
print(f"解像度: {capture.get(cv2.CAP_PROP_FRAME_WIDTH)}×{capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

print("カメラを起動しました")
print("目線をカメラに向け、Enterキーを押してください")
while True:
    ret, frame = capture.read()
    cv2.imshow("frame", frame)
    dets = detector(frame[:, :, ::-1])
    if len(dets) > 0:
        parts = predictor(frame, dets[0]).parts()
        key = cv2.waitKey(1)
        if key == 13:  # Enterキー
            global def_Right_eye_pupil_center, def_Left_eye_pupil_center
            Right_eye_pupil_center_x, Right_eye_pupil_center_y, Left_eye_pupil_center_x, Left_eye_pupil_center_y = show_eye(frame, parts)
            def_Right_eye_pupil_center = (Right_eye_pupil_center_x, Right_eye_pupil_center_y)
            def_Left_eye_pupil_center = (Left_eye_pupil_center_x, Left_eye_pupil_center_y)

            break

print("正面の目線を記憶しました")
print("----------視線推定を開始します----------")

while True:
    # カメラ映像の受け取り
    ret, frame = capture.read()

    # detetorによる顔の位置の推測
    dets = detector(frame[:, :, ::-1])
    if len(dets) > 0:
        # predictorによる顔のパーツの推測
        parts = predictor(frame, dets[0]).parts()
        frame_copy = frame.copy()
        # def_Right_eye_pupil_centerとのずれを表示
        x_r, x_l, y_t, y_b = set_xy(x_max, x_min, y_max, y_min)
        Right_eye_pupil_center_x, Right_eye_pupil_center_y, Left_eye_pupil_center_x, Left_eye_pupil_center_y = show_eye(frame, parts, xy=[x_r, x_l, y_t, y_b])
        Right_eye_pupil_center = (Right_eye_pupil_center_x, Right_eye_pupil_center_y)
        Left_eye_pupil_center = (Left_eye_pupil_center_x, Left_eye_pupil_center_y)
        # 右目のずれ
        if Right_eye_pupil_center_x != None and Right_eye_pupil_center_y != None:
            cv2.line(frame_copy, def_Right_eye_pupil_center, Right_eye_pupil_center, (0, 0, 255), thickness=3, lineType=cv2.LINE_4)
            cv2.putText(frame_copy, f"Right_diff : {(def_Right_eye_pupil_center[0]-Right_eye_pupil_center_x, def_Right_eye_pupil_center[1]-Right_eye_pupil_center_y)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        # 左目のずれ
        if Left_eye_pupil_center_x != None and Left_eye_pupil_center_y != None:
            cv2.line(frame_copy, def_Left_eye_pupil_center, Left_eye_pupil_center, (0, 0, 255), thickness=3, lineType=cv2.LINE_4)
            cv2.putText(frame_copy, f"Left_diff : {(def_Left_eye_pupil_center[0]-Left_eye_pupil_center_x, def_Left_eye_pupil_center[1]-Left_eye_pupil_center_y)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    cv2.imshow("frame", frame_copy)

    key = cv2.waitKey(1)
    # エスケープキーを押して終了します
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
        