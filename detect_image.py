
from ultralytics import YOLO

# 1. 加载 YOLO 预训练模型
model = YOLO("yolo26n.pt")

# 2. 输入测试图片，进行目标检测
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True,
    conf=0.25
)

# 3. 输出检测结果
print("检测完成！")
print("检测到的目标数量：", len(results[0].boxes))

