# 从model模块导入SimpleNet类
from model.simple_nn import SimpleNet

# 导入PyTorch库
import torch

# 设置保存模型参数的文件路径
path_state_dict = './4_simple_nn_model.pth'

# 创建SimpleNet类的实例，这会初始化模型
model = SimpleNet()

# 加载保存的模型参数到模型实例
model.load_state_dict(torch.load(path_state_dict))

# 将模型切换到评估模式，这对于进行预测是必要的，因为它会禁用一些特定于训练阶段的操作，比如Dropout
model.eval()

# 生成随机输入数据：1个样本，每个样本10个特征
input_data = torch.randn(1, 784)

# 使用模型进行预测，此时不计算梯度以提高性能和减少内存使用
with torch.no_grad():
    output = model(input_data)
