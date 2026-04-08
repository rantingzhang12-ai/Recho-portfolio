# 交通标志识别系统

基于PyTorch和ResNet18的交通标志识别系统，支持训练和实时摄像头推理。

## 功能特性

- 支持30种常见交通标志的识别
- 使用预训练ResNet18进行迁移学习
- 包含完整的数据增强和预处理流程
- 实时摄像头推理功能
- 置信度阈值过滤
- FPS显示

## 安装依赖
bash

pip install -r requirements.txt

## 数据集准备

1. 下载数据集并解压到项目目录
2. 确保目录结构如下：
archive_traffic/

├── train/

│ ├── class1/

│ ├── class2/

│ └── ...

└── valid/

├── class1/

├── class2/

└── ...

## 训练模型
bash

python traffic_model.py

训练参数可在`Config`类中调整：
- 学习率(LR)
- 批次大小(BATCH_SIZE)
- 训练轮数(EPOCHS)
- 图像尺寸(IMG_SIZE)

## 实时推理
bash

python traffic_test.py

使用说明：
- 按`q`键退出
- 置信度阈值可在`Config`类中调整

## 模型结构
TrafficSignModel(

(backbone): ResNet18

(classifier): Sequential(

(0): Linear(in_features=512, out_features=256, bias=True)

(1): ReLU()

(2): Dropout(p=0.5, inplace=False)

(3): Linear(in_features=256, out_features=30, bias=True)

)

)

## 性能指标

在验证集上：
- 准确率: >98%
- 推理速度: ~30FPS (GTX 1080Ti)

## 已知类别

1. 限速30
2. 限速60
3. 禁止直行左转
4. 禁止直行
5. 禁止左转
6. 禁止超车
7. 禁止掉头
8. 禁止行车
9. 禁止鸣笛
10. 可直行右转
11. 只能直行
12. 只能左转
13. 只能左转右转
14. 环岛路
15. 只能小车行驶
16. 可鸣笛
17. 可自行车行驶
18. 可掉头
19. 前方障碍两侧通行
20. 注意行人
21. 前方学校
22. 向右急转弯
23. 下陡坡
24. 慢行
25. T形路口
26. 反向弯路
27. 前方施工
28. 停车让行
29. 禁止停车
30. 减速让行

## 注意事项

1. 确保摄像头权限已开启
2. 推理时不要在光照条件很差的时候进行
3. 标志应尽量正对摄像头
4. 训练数据不足可能导致某些类别识别效果不佳