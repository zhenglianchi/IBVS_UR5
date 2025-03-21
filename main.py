import numpy as np
import cv2
import time
import pyrealsense2 as rs
from Servo import servo
from UR_Base import UR_BASE
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


def resize_and_center_box(target_points, image_size, padding=0):
    # 计算目标框的中心点
    center_x = np.mean([point[0] for point in target_points])
    center_y = np.mean([point[1] for point in target_points])

    # 图像中心点
    image_center_x = image_size[0] / 2
    image_center_y = image_size[1] / 2

    # 计算目标框与图像中心的偏移量
    offset_x = image_center_x - center_x
    offset_y = image_center_y - center_y

    # 将目标框移动到图像中心
    moved_points = [[point[0] + offset_x, point[1] + offset_y] for point in target_points]

    # 计算移动后的目标框的宽度和高度
    x_coords, y_coords = zip(*moved_points)
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)

    # 计算放大比例
    max_dim = max(width, height)
    scale_factor = (max_dim + 2 * padding) / max_dim

    # 等比例放大目标框
    scaled_points = [[int((point[0] - image_center_x) * scale_factor + image_center_x),
                      int((point[1] - image_center_y) * scale_factor + image_center_y)] for point in moved_points]

    return scaled_points

def get_aligned_images():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    aligned_color_frame = aligned_frames.get_color_frame()
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics
    img_color = np.asanyarray(aligned_color_frame.get_data())
    img_depth = np.asanyarray(aligned_depth_frame.get_data())
    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame




if __name__ == "__main__":
    global img_color

    HOST = '192.168.111.10'

    # 初始化UR5机械位姿
    first_tcp = np.array([-0.27829, -0.11943, 0.49383, 1.148, 2.447, -2.557])

    ur5 = UR_BASE(HOST,first_tcp)
    
    # 加载 YOLO 模型（确保 yolo11n.pt 模型文件在正确路径下）
    #model = YOLO("yolo11n.pt").to("cuda")
    model = YOLO("yolo11n.pt")

    time.sleep(2)

    print("初始位姿:",ur5.get_tcp())

    # 控制增益
    lambda_gain = 0.03

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # 定义全局变量，存储鼠标点击位置的坐标和半径
    target_point = None

    start_time = time.time()  # 获取程序开始时间

    detected_points = None

    target_points = None

    center_point = None

    cv2.namedWindow("RealSence")

    while True:
        color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()

        f = [color_intrin.fx,color_intrin.fy]
        resolution = [color_intrin.width,color_intrin.height]

        # 调用 YOLO 模型进行检测
        results = model(img_color)
        # 获取检测结果
        # 每个检测框数据格式为 [x1, y1, x2, y2, confidence, class_id]
        boxes = results[0].boxes.data.cpu().numpy()

        # 遍历每个检测框
        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = box

            if cls_id == 67 and conf > 0.5:
                # 转换坐标为整数
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # 绘制矩形框（颜色为绿色，线宽为2）
                cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 生成标签文本（类别和置信度）
                label = f"{names[int(cls_id)]} {conf:.2f}"
                # 绘制标签（在框上方显示）
                cv2.putText(img_color, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detected_points = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
                # 计算所有点的 x 坐标和 y 坐标的平均值
                average_x = (detected_points[0][0] + detected_points[1][0] + detected_points[2][0] + detected_points[3][0]) / 4
                average_y = (detected_points[0][1] + detected_points[1][1] + detected_points[2][1] + detected_points[3][1]) / 4

                # 得到中心点坐标
                center_point = (average_x, average_y)

                target_points = resize_and_center_box(detected_points,resolution)

                for point in target_points:
                    cv2.circle(img_color, point, 3, (255, 255, 255), -1)

                uv = np.array(detected_points).T
                p_star = np.array(target_points).T

                print("检测点为:\n",uv)
                print("目标点为:\n",p_star)
                
                try:
                    servo(ur5,uv,img_depth,p_star,lambda_gain,f,resolution,center_point)
                except:
                    ur5.disconnect()
                    pipeline.stop()
                    cv2.destroyAllWindows()
            else:
                detected_points = None

        
            
        cv2.imshow('RealSence', img_color)
        key = cv2.waitKey(1)
        if key == 27 :
            ur5.disconnect()
            pipeline.stop()
            cv2.destroyAllWindows()
            break




