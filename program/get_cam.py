import cv2

# VideoCaptureオブジェクトの作成 
capture = cv2.VideoCapture(0) # 複数のカメラが接続されている場合は引数で指定できます

# 解像度の変更
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
print(f"解像度: {capture.get(cv2.CAP_PROP_FRAME_WIDTH)}×{capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

while(True):
    ret, frame = capture.read()
    cv2.imshow('frame',frame)
    # エスケープキーが押されたら終了
    if cv2.waitKey(1) == 27:
        break

# カメラを終了しウィンドウを閉じる
capture.release()
cv2.destroyAllWindows()
