import cv2

# mostra o fps, bitrate e a quantidade de frames de um video e todos as possiveis propriedades dele
video_path = "D:/Projects_Datasets/EyeTracker/dataset/recordings/processed/2/18/video.avi"
print(f'video_path: {video_path}')
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
bitrate = cap.get(cv2.CAP_PROP_BITRATE)
print("fps: ", fps)
print("total_frames: ", total_frames)
print("bitrate: ", bitrate)
print("width: ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("height: ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("exposure: ", cap.get(cv2.CAP_PROP_EXPOSURE))
print("convert_rgb: ", cap.get(cv2.CAP_PROP_CONVERT_RGB))
print("rectification: ", cap.get(cv2.CAP_PROP_RECTIFICATION))
print("iso_speed: ", cap.get(cv2.CAP_PROP_ISO_SPEED))
print("buffersize: ", cap.get(cv2.CAP_PROP_BUFFERSIZE))
print("mode: ", cap.get(cv2.CAP_PROP_MODE))

