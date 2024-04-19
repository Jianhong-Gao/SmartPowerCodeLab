import torch
from torchvision import transforms
from model.simple_nn import SimpleNet

class SimpleNet_Predictor:
    def __init__(self, model_path):
        """
        初始化模型预测器，加载模型并进行热身。
        :param model_class: 模型类，用于实例化模型
        :param model_path: 预训练模型的路径，用于加载模型参数
        """
        # 实例化模型
        self.model = SimpleNet()
        # 加载模型参数
        self.model.load_state_dict(torch.load(model_path))
        # 切换到评估模式
        self.model.eval()
        # 定义数据预处理
        self.transform = transforms.Compose([
              # 此处为空，可自定义
        ])
        # 进行模型热身，以确保模型在首次调用时响应迅速
        self.warmup()

    def warmup(self):
        """
        执行一次前向传递以热身模型。
        """
        with torch.no_grad():
            # 使用一些随机数据进行热身
            random_input = torch.randn(1, 784)
            random_input = self.transform(random_input)  # 应用转换
            self.model(random_input)

    def __call__(self, input_data):
        """
        使得该类实例可以像函数那样被调用，进行模型预测。
        :param input_data: 输入数据，应为torch.Tensor格式
        :return: 模型的输出结果
        """
        with torch.no_grad():
            input_data = self.transform(input_data)  # 应用转换
            return self.model(input_data)



