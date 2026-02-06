深度学习目标检测算法yolo训练无人机目标检测_水上漂浮垃圾数据集识别水面上的漂浮物及垃圾进行智能识别
@[toc]
文章内容及代码仅供参考。

水上漂浮垃圾数据集 1类别YOLO格式｜水域环保专用

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2837eb0ba64e4226beb571bfe00ab5f2.png)


①1501张图片（训练集1050张，验证集300张，测试集151张）
②标注，YOLO格式
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5236595f46534f25a2ffcaec0137dff3.png)

应用场景​：
①水域环保：水上垃圾自动识别与清理
②智能监测：水域垃圾实时监控与预警
③自动化清理系统：提升效率，降低成本
④视觉导航：无人船或无人机目标检测

数据集
标注格式：txt（YOLO格式）
类别标签：litter（水上垃圾检测，8181个框）
数据来源：真实水域场景采集

以下是一个详细的流程，从数据集准备、模型训练到构建 GUI 应用程序的完整实现。我们将使用 YOLOv8 作为示例模型，并附上每一步的详细代码。

---

## **一、安装依赖**

确保已安装 Python（推荐版本 3.8+），并安装以下依赖：

```bash
pip install torch torchvision ultralytics opencv-python matplotlib pyqt5
```

- `torch` 和 `torchvision`：深度学习框架。
- `ultralytics`：YOLOv8 的官方库。
- `opencv-python`：图像处理。
- `matplotlib`：可视化工具。
- `pyqt5`：用于构建 GUI 应用程序。

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d1a6114b9d174b66bb860d260c36301f.png)

## **二、准备数据集**

### **1. 数据集目录结构**
将水上漂浮垃圾数据集按照以下格式组织：

```
floating-garbage-dataset/
├── images/
│   ├── train/          # 训练集图片
│   ├── val/            # 验证集图片
│   └── test/           # 测试集图片
├── labels/
│   ├── train/          # 训练集标注文件（txt 格式）
│   ├── val/            # 验证集标注文件（txt 格式）
│   └── test/           # 测试集标注文件（txt 格式）
└── data.yaml           # 数据集配置文件
```

### **2. 创建 `data.yaml`**
在 `floating-garbage-dataset/` 目录下创建 `data.yaml` 文件，内容如下：

```yaml
train: ./images/train
val: ./images/val
test: ./images/test

nc: 1                   # 类别数量
names: ['litter']       # 类别名称
```

### **3. 检查数据集**
确保所有图片和对应的 `.txt` 标注文件都正确放置在各自目录中。`.txt` 文件的标注格式为：

```
<class_id> <x_center> <y_center> <width> <height>
```

例如：
```
0 0.5 0.5 0.2 0.3
```

---

## **三、训练模型**

### **1. 开始训练**
运行以下代码开始训练模型：

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')  # 使用 YOLOv8 Nano 模型

# 训练模型
results = model.train(
    data='path/to/floating-garbage-dataset/data.yaml',  # 数据集配置文件路径
    epochs=100,                                # 训练轮数
    imgsz=640,                                 # 输入图片尺寸
    batch=16,                                  # 批量大小
    device='cuda' if torch.cuda.is_available() else 'cpu'  # 使用 GPU 或 CPU
)

# 导出模型（可选）
model.export(format='onnx')  # 导出为 ONNX 格式
```

---

## **四、评估模型**

### **1. 测试模型性能**
运行以下代码进行模型评估：

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')

# 测试模型
metrics = model.val(data='path/to/floating-garbage-dataset/data.yaml')
print(f"mAP50: {metrics.box.map50}")  # 输出 mAP50 指标
```

---

## **五、构建 GUI 应用程序**

以下是一个完整的 PyQt5 GUI ，用于加载模型并检测图片中的水上漂浮垃圾目标。仅供参考。同学

```python
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO

class FloatingGarbageDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        # 加载训练好的模型
        self.model = YOLO('runs/detect/train/weights/best.pt')

    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle("水上漂浮垃圾检测")
        self.setGeometry(100, 100, 800, 600)

        # 创建一个 QLabel 用于显示图片
        self.label = QLabel(self)
        self.label.setText("选择图片以检测水上漂浮垃圾")
        self.label.setAlignment(Qt.AlignCenter)

        # 创建一个按钮用于选择图片
        self.button = QPushButton("选择图片", self)
        self.button.clicked.connect(self.load_image)

        # 布局设置
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        # 打开文件对话框选择图片
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            # 读取图片
            image = cv2.imread(file_path)
            if image is None:
                print("无法读取图片，请检查路径是否正确！")
                return

            # 使用模型进行目标检测
            results = self.model(image)  # 进行目标检测
            annotated_image = results[0].plot()  # 绘制检测结果

            # 将 OpenCV 图像转换为 PyQt 图像
            height, width, channel = annotated_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(annotated_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # 显示图片
            self.label.setPixmap(QPixmap.fromImage(q_img))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FloatingGarbageDetectorApp()
    window.show()
    sys.exit(app.exec_())
```

---

## **六、运行应用程序**

1. 将上述代码保存为 `app.py`。
2. 确保已安装所有依赖库。
3. 在终端中运行以下命令启动 GUI 应用程序：

```bash
python app.py
```

---

## **七、总结**

通过以上步骤，可万成 水上漂浮垃圾检测数据集的准备、模型训练、评估以及构建一个简单的 GUI 应用程序来展示模型的检测效果。每个步骤的关键点总结：

1. **数据集准备**：
   - 确保图片和标注文件格式正确。
   - 创建 `data.yaml` 文件定义类别和数据路径。

2. **模型训练**：
   - 使用 YOLOv8 提供的简单接口进行训练。
   - 可根据需要调整超参数（如 `epochs`、`batch`、`imgsz`）。

3. **模型评估**：
   - 使用 `val()` 方法评估模型性能，重点关注 `mAP50` 指标。

4. **GUI 应用程序**：
   - 使用 PyQt5 构建一个简单的界面，用户可以选择图片并查看检测结果。

仅供参考，我的同学们。





