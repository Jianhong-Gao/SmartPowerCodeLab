from utils_AI.simplenet_predictor import SimpleNet_Predictor
import torch

model_path = './4_simple_nn_model.pth'  # 假设模型参数已经保存在这个路径
predictor = SimpleNet_Predictor(model_path)

# 测试模型预测
test_input = torch.randn(1, 784)
output = predictor(test_input)
print("Model output shape:", output.shape)