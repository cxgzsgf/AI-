# 第一次代码使用请阅读README文件！
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
# 1. 加载模型
def load_model():
    choice = input("请选择模型类型:\n1. 使用预训练模型\n2. 使用自己的模型\n输入1 或 2: ")#加载预训练模型或自定义模型。
    if choice == '1':
        model_name = 'yolov8n.pt'  # 预训练模型
        print(f"加载预训练模型: {model_name}")
    elif choice == '2':
        model_name = input("请输入自定义模型的路径（例如: path/to/your_model.pt）: ")
        print(f"加载自定义模型: {model_name}")
    else:
        print("无效选项，默认加载预训练模型 yolov8n.pt")
        model_name = 'yolov8n.pt'
    model = YOLO(model_name)
    return model
# 2. 训练模型
def train_model(model, data_path='coco.yaml', epochs=10):
    print("开始训练...")
    results = model.train(data=data_path, epochs=epochs, imgsz=640, lr0=0.01, batch=16)
    print("训练完成！")
    return results
# 3. 批量检测数据集图片
def detect_images_in_directory(model, directory_path):
    for filename in os.listdir(directory_path):
        image_path = os.path.join(directory_path, filename)

        if filename.endswith(('.jpg', '.jpeg', '.png')):
            print(f"开始检测图片：{image_path}")
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = model.predict(source=image_rgb, save=False, verbose=False)
            print("图片检测完成！")
            display_results(image_rgb, results)
# 4. 视频检测
def detect_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件：{video_path}")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, save=False, verbose=False)
        for result in results:
            boxes = result.boxes
            names = result.names
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = box.conf[0].item()
                    bbox = box.xyxy[0].tolist()
                    label = names[cls_id]
                    text = f"{label} {conf:.2f}"
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("Video Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
# 5. 显示检测结果（图片）
def display_results(image, results):
    for result in results:
        boxes = result.boxes
        names = result.names
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = box.conf[0].item()
                bbox = box.xyxy[0].tolist()
                label = names[cls_id]
                text = f"{label} {conf:.2f}"
                x_min, y_min, x_max, y_max = map(int, bbox)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
# 6. 主代码
if __name__ == '__main__':
    model = load_model()
    # 选择否需要训练
    train_choice = input("是否需要训练模型? (y/n): ")
    if train_choice.lower() == 'y':
        # 训练次数
        epochs = int(input("请输入训练轮数: "))
        # 训练模型
        train_results = train_model(model, epochs=epochs)
    # 选择检测模式
    detect_choice = input("请选择检测模式:\n1. 图片检测\n2. 视频检测\n输入选项 1 或 2: ")
    if detect_choice == '1':
        # 图片检测示例
        dataset_directory = 'C:/Users/1/Desktop/pythonProject/data/coco/images/val2017'#使用coco的检测图片
        detect_images_in_directory(model, dataset_directory)
    elif detect_choice == '2':
        # 视频检测示例
        video_path = input("请输入视频文件路径: ")
        detect_video(model, video_path)
    else:
        print("无效选项")
