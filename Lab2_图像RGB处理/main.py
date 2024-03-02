import cv2
import numpy as np
from PIL import Image

# 定义函数来计算图像的平均亮度
def calculate_brightness(image_tensor):
    # 获取图像的通道数
    channels = image_tensor.shape[2]
    
    # 初始化存储每个通道平均亮度的列表
    avg_brightness = []
    
    # 计算每个通道的平均亮度
    for i in range(channels):
        channel_mean = np.mean(image_tensor[:, :, i])
        avg_brightness.append(channel_mean)
    
    return avg_brightness

# 加载图像并转换为张量（使用OpenCV）
def load_image_with_cv2(image_path):
    # 使用OpenCV读取图像
    image = cv2.imread(image_path)
    # 将图像从BGR格式转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 将图像转换为张量
    image_tensor = np.array(image_rgb)
    return image_tensor

# 加载图像并转换为张量（使用PIL）
def load_image_with_pil(image_path):
    # 使用PIL库打开图像
    image = Image.open(image_path)
    # 将图像转换为张量
    image_tensor = np.array(image)
    return image_tensor

# 图像路径
image_path = "./images/colorful.jpg"

# 使用OpenCV加载图像并计算平均亮度
image_cv2 = load_image_with_cv2(image_path)
brightness_cv2 = calculate_brightness(image_cv2)
print("使用OpenCV加载的图像平均亮度：", brightness_cv2)

# 使用PIL加载图像并计算平均亮度
image_pil = load_image_with_pil(image_path)
brightness_pil = calculate_brightness(image_pil)
print("使用PIL加载的图像平均亮度：", brightness_pil)

# 根据平均亮度判断图像颜色
threshold = 111  # 设置阈值

if all(brightness_cv2[i] > threshold for i in range(3)):
    print("这是一张红色图像")
elif all(brightness_cv2[i] < threshold for i in range(3)):
    print("这是一张蓝色图像")
elif brightness_cv2[0] < threshold and brightness_cv2[1] > threshold and brightness_cv2[2] < threshold:
    print("这是一张绿色图像")
else:
    print("这张图像有多种颜色")

