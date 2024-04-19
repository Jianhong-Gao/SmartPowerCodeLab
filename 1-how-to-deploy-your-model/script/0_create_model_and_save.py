# 从 model 模块中导入 SimpleNet 类
from model.simple_nn import SimpleNet

# 导入 PyTorch 库
import torch

# 创建 SimpleNet 类的实例，这会初始化模型
model = SimpleNet()

# 生成随机输入数据：64个样本，每个样本10个特征
inputs = torch.randn(64, 784)

# 将输入数据传递给模型，进行一次前向计算，得到输出
output = model(inputs)
# 设置保存模型参数的文件路径
path_state_dict = './4_simple_nn_model.pth'
# 保存模型的参数到文件 'simple_nn_model.pth'
torch.save(model.state_dict(), path_state_dict)

# 打印消息确认模型参数已经被保存
print("Model parameters saved to '4_simple_nn_model.pth'")
