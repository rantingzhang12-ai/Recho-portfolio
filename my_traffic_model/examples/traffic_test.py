import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import torchvision.models as models


# ======================
# 1. 配置参数
# ======================
class Config:
    MODEL_PATH = "../models/final_traffic_sign_model.pth"
    IMG_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONFIDENCE_THRESHOLD = 0.7


# ======================
# 2. 类别名称映射
# ======================
CLASS_NAMES = [
    '限速30', '限速60', '禁止直行左转', '禁止直行', '禁止左转', '禁止超车',
    '禁止掉头', '禁止行车', '禁止鸣笛', '可直行右转', '只能直行', '只能左转',
    '只能左转右转', '环岛路', '只能小车行驶', '可鸣笛', '可自行车行驶',
    '可掉头', '前方障碍两侧通行', '注意行人', '前方学校', '向右急转弯',
    '下陡坡', '慢行', 'T形路口', '反向弯路', '前方施工', '停车让行',
    '禁止停车', '减速让行'
]
assert len(CLASS_NAMES) == 30, f"类别数量不匹配，预期30类，实际{len(CLASS_NAMES)}类"
# assert 确保类别数量匹配，防止索引越界错误

# ======================
# 3. 加载模型 (与训练脚本完全匹配)
# ======================
def load_model(model_path, num_classes=30):
    """
    加载训练好的模型(与训练时结构完全一致)
    """
    # 使用与训练时相同的模型定义：模型初始化
    model = TrafficSignModel(num_classes)

    # 加载权重(使用strict=False忽略不匹配的键)
    # weights_only=True：防止恶意pickle数据注入（安全特性）
    state_dict = torch.load(model_path, map_location=Config.DEVICE, weights_only=True)

    # 将 backbone 和 classifier 的权重加载到当前模型中
    backbone_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
    model.backbone.load_state_dict(backbone_state_dict, strict=False)

    classifier_state_dict = {k.replace('classifier.', ''): v for k, v in state_dict.items() if
                             k.startswith('classifier.')}
    model.classifier.load_state_dict(classifier_state_dict, strict=False)

    # 4. 移动模型到设备并设置为评估模式
    # 将模型移动到GPU/CPU
    model.to(Config.DEVICE)
    model.eval()   #关闭Dropout和BatchNorm的随机性，确保推理一致性。
    return model


class TrafficSignModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


model = load_model(Config.MODEL_PATH, num_classes=30)


# ======================
# 4. 图像预处理
# ======================
def preprocess_image(image):
    """
    预处理输入图像(需与训练时的预处理一致)
    """
    # 1. BGR转RGB
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 2. 定义预处理流程
    transform = transforms.Compose([
        transforms.Resize(256),                     # 缩放短边至256
        transforms.CenterCrop(Config.IMG_SIZE),     # 中心裁剪至224x224
        transforms.ToTensor(),                      # 转张量并归一化到[0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor.to(Config.DEVICE)    #预处理后的张量直接移动到模型所在设备（GPU/CPU）


# ======================
# 5. 推理函数
# ======================
def predict(image_tensor, model, class_names):
    """
    执行推理
    """
    with torch.no_grad():    # 禁用梯度计算
        outputs = model(image_tensor)       # 前向传播
        probabilities = torch.nn.functional.softmax(outputs, dim=1)     # 计算概率分布
        confidence, pred_class = torch.max(probabilities, dim=1)        # 选择最高置信度类别

    confidence = confidence.item()
    pred_class = pred_class.item()

    # 索引越界检查
    if pred_class >= len(class_names):
        raise ValueError(f"Predicted class index {pred_class} is out of range for {len(class_names)} classes.")

    pred_label = class_names[pred_class]
    print("Predict 返回值:", pred_label)  # 打印返回值
    return pred_class, confidence, pred_label


# ======================
# 6. 实时摄像头推理
# ======================
def real_time_inference(class_names):
    """
    实时摄像头推理
    """
    # 1. 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头!")
        return

    print("按'q'键退出...")
    prev_time = 0

    # 3. 主循环
    while True:
        # 3.1 读取帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧!")
            break

        # 3.2 计算FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)  #得到上一帧的处理耗时
        prev_time = curr_time

        # 3.3 预处理图像
        input_tensor = preprocess_image(frame)
        # 3.4 模型推理
        pred_class, confidence, pred_label = predict(input_tensor, model, class_names)

        # 3.5 可视化结果(仅当置信度高于阈值时)
        if confidence >= Config.CONFIDENCE_THRESHOLD:
            label = f"{pred_label}: {confidence:.2f}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 3.6 显示FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # 3.7 显示图像
        cv2.imshow("Trafic Sign Recognition", frame)

        # 3.8 退出条件
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ======================
# 7. 主函数
# ======================
if __name__ == "__main__":
    real_time_inference(CLASS_NAMES)
