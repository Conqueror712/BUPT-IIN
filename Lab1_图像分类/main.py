import torch
from torchvision import models, transforms
from PIL import Image

# 本地图像文件路径
img_path = "./images/demo1.jpg"

# 加载示例图像
img = Image.open(img_path)

# 加载预训练的AlexNet模型
model = models.alexnet(pretrained=True)
model.eval()

# 打印网络结构
print(model)

# 定义数据转换
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 对图像进行预处理
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)

# 将模型移到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_batch = input_batch.to(device)

# 执行预测
with torch.no_grad():
    output = model(input_batch)

# 获取预测结果
probabilities = torch.nn.functional.softmax(output[0], dim=0)
class_idx = torch.argmax(probabilities).item()

# 加载类别标签
with open("./imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# 打印分类结果
print("Predicted class:", classes[class_idx], ", Probability:", probabilities[class_idx].item())
