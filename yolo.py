import cv2
from ultralytics import YOLO

names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# 加载 YOLO 模型（确保 yolo11n.pt 模型文件在正确路径下）
model = YOLO("yolo11n.pt").to("cuda")

# 打开摄像头（参数0表示默认摄像头）
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取视频帧")
        break

    # 若颜色有问题，可以转换为 RGB 格式：
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame.copy()

    # 调用 YOLO 模型进行检测
    results = model(img)
    # 获取检测结果（假设检测结果在 results[0] 中）
    # 每个检测框数据格式为 [x1, y1, x2, y2, confidence, class_id]
    boxes = results[0].boxes.data.cpu().numpy()

    # 遍历每个检测框
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box

        if cls_id == 41:
            # 转换坐标为整数
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # 绘制矩形框（颜色为绿色，线宽为2）
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 生成标签文本（类别和置信度）
            label = f"{names[int(cls_id)]} {conf:.2f}"
            # 绘制标签（在框上方显示）
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # 打印目标框四个角的坐标：左上、右上、右下、左下
            print(f"目标: {names[int(cls_id)]}, 四个角坐标: "
                f"({x1}, {y1}), ({x2}, {y1}), ({x2}, {y2}), ({x1}, {y2})")
        else:
            continue

    # 显示带检测框的摄像头画面
    cv2.imshow("YOLO 实时检测", frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
