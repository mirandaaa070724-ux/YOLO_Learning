import os
import cv2
from ultralytics import YOLO
from tkinter import Tk, filedialog

# 1. 加载 YOLO 模型
model = YOLO("yolo26n.pt")

# 2. 选择视频
root = Tk()
root.withdraw()

video_path = filedialog.askopenfilename(
    title="请选择需要检测的视频",
    filetypes=[
        ("视频文件", "*.mp4 *.avi *.mov *.mkv"),
        ("所有文件", "*.*")
    ]
)

root.destroy()

if not video_path:
    print("没有选择视频！")
    exit()

# 3. 使用 OpenCV 打开视频
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("视频打开失败！")
    exit()

# 4. 获取视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

delay = max(1, int(1000 / fps))

print("视频帧率：", fps)
print("开始目标检测，按 P 暂停，按 Q 退出")

output_path = os.path.join(
    os.path.dirname(video_path),
    "yolo_result.mp4"
)

# 设置视频编码格式
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# 创建视频写入对象
out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
)

print("输出视频路径：", output_path)

# 5. 逐帧读取和检测
paused = False

while cap.isOpened():

    if not paused:
        # 读取一帧
        ret, frame = cap.read()

        if not ret:
            print("视频播放结束！")
            break

        # YOLO 对当前帧进行检测
        results = model.predict(
            source=frame,
            verbose=False
        )

        # 绘制检测框
        annotated_frame = results[0].plot()

        # 将当前检测画面写入输出视频
        out.write(annotated_frame)

        # OpenCV 显示检测结果
        # 获取画面尺寸
        h, w = annotated_frame.shape[:2]

        # 设置窗口最大尺寸
        max_width = 1200
        max_height = 750

        # 计算缩放比例，保持原始宽高比
        scale = min(max_width / w, max_height / h, 1)

        # 缩小画面
        display_frame = cv2.resize(
            annotated_frame,
            (int(w * scale), int(h * scale))
        )

        # 显示缩小后的画面
        cv2.imshow("YOLO Object Detection", display_frame)

    # 6. 获取键盘输入
    key = cv2.waitKey(30 if paused else delay) & 0xFF

    if key == ord('q'):
        print("退出播放！")
        break

    elif key == ord('p'):
        paused = not paused

# 7. 释放资源
cap.release()
# 释放视频写入对象
out.release()
cv2.destroyAllWindows()

print("程序结束！")
