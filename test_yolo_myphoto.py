from ultralytics import YOLO
from tkinter import Tk, filedialog

# 1. 加载模型
model = YOLO("yolo26n.pt")

# 2. 对图片进行检测
# 创建文件选择窗口
root = Tk()
root.withdraw()

# 选择图片
image_path = filedialog.askopenfilename(
    title="请选择一张图片",
    filetypes=[
        ("图片文件", "*.jpg *.jpeg *.png *.bmp"),
        ("所有文件", "*.*")
    ]
)

# 如果没有选择图片，就退出程序
if not image_path:
    print("没有选择图片，程序退出！")
    exit()

# 对输入的图片进行检测
results = model.predict(source=image_path)

# 3. 查看返回结果的类型
print(type(results))

# 4. 查看检测结果数量
print(len(results))

# 5. 获取第一张图片的检测结果
result = results[0]

# 6. 查看结果对象的类型
print(type(result))

# 7. 查看检测框信息
print(result.boxes)

# 8. 查看类别名称
print(result.names)
# 遍历每一个检测框
for box in result.boxes:

    # 获取类别编号
    cls_id = int(box.cls[0])

    # 根据类别编号获取名称
    class_name = result.names[cls_id]

    # 获取置信度
    confidence = float(box.conf[0])

    # 获取矩形框坐标
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    print("类别：", class_name)
    print("置信度：", confidence)
    print("位置：", x1, y1, x2, y2)
    print("--------------------")
    # 创建一个空字典
object_count = {}

# 遍历所有检测框
for box in result.boxes:

    # 获取类别编号
    cls_id = int(box.cls[0])

    # 获取类别名称
    class_name = result.names[cls_id]

    # 如果类别已经存在，数量加1
    if class_name in object_count:
        object_count[class_name] += 1

    # 如果类别不存在，初始化为1
    else:
        object_count[class_name] = 1

# 输出统计结果
print("\n检测结果统计：")

for name, count in object_count.items():
    print(f"{name}: {count}")

print("总目标数量：", sum(object_count.values()))
# 绘制检测结果
annotated_image = result.plot()

# 保存检测后的图片
result.save(filename="result.jpg")

print("检测结果图片已保存！")