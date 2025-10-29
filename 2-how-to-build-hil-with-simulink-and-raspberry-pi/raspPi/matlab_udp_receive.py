import socket
import struct
import matplotlib.pyplot as plt
import os

class UDPADCReceiver:
    """
    模拟 ADC 数据的 UDP 接收类。
    可自动忽略初始化零包，从第一个有效数据包开始计数。
    """

    def __init__(self, recv_port=5005, fs=5000, interval=0.5, adc_range=10.0, data_type='int16'):
        """
        初始化接收器配置
        """
        self.UDP_IP = '0.0.0.0'
        self.RECV_PORT = recv_port
        self.fs = fs
        self.interval = interval
        self.adc_range = adc_range
        self.data_type = data_type

        # int16 -> 2字节
        self.bytes_per_sample = 2 if data_type == 'int16' else 8
        self.samples_per_packet = int(fs * interval)
        self.expected_packet_size = self.samples_per_packet * self.bytes_per_sample

        self.packet_count = 0
        self.first_valid_found = False

        # UDP 初始化
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.UDP_IP, self.RECV_PORT))
        print(f"✅ 已启动 UDP 监听：端口 {self.RECV_PORT}")
        print(f"期望接收包大小：{self.expected_packet_size} 字节（{self.samples_per_packet} 个采样点）")

        # 创建输出文件夹
        os.makedirs('received_data_images', exist_ok=True)

    def recv_packet(self):
        """
        接收一个 UDP 数据包并返回有效波形（单位：V）
        """
        data, addr = self.sock.recvfrom(self.expected_packet_size + 200)
        packet_length = len(data)
        print(f"接收到的数据包长度：{packet_length} 字节")

        if packet_length != self.expected_packet_size:
            print(f"⚠️ 警告：接收到的包长度不符合预期 ({packet_length}/{self.expected_packet_size})")
            return None

        # 按 int16 解包
        fmt = f'{self.samples_per_packet}h'
        samples = struct.unpack(fmt, data)

        # 忽略全零包
        if max(samples) == 0 and min(samples) == 0:
            print("⚠️ 检测到全零包，忽略。")
            return None

        # 从第一个有效包开始计数
        if not self.first_valid_found:
            self.first_valid_found = True
            self.packet_count = 1
            print("✅ 检测到第一个有效数据包，计数从 1 开始。")
        else:
            self.packet_count += 1

        voltages = [x / 32767.0 * self.adc_range for x in samples]
        print(f"✅ 第 {self.packet_count} 个有效数据包，采样点数：{self.samples_per_packet}")
        return voltages

    def save_waveform(self, voltages):
        """
        保存电压波形图为 PNG 文件
        """
        filename = f'received_data_images/packet_{self.packet_count}.png'
        plt.figure(figsize=(10, 4))
        plt.plot(voltages, color='tab:blue', linewidth=1)
        plt.xlabel('Sample Index')
        plt.ylabel('Voltage (V)')
        plt.title(f'ADC Waveform - Packet {self.packet_count}')
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f"💾 图像已保存至：{filename}")

    def run(self):
        """
        主循环：持续接收数据并绘图保存
        """
        print("🟢 开始接收 UDP 数据...\n")
        while True:
            voltages = self.recv_packet()
            if voltages is not None:
                self.save_waveform(voltages)

# ====================== 主程序入口 ======================
if __name__ == "__main__":
    receiver = UDPADCReceiver(
        recv_port=5005,  # 端口号
        fs=5e3,          # 采样率
        interval=0.5,    # 每包时间
        adc_range=10.0,  # 量程 ±10V
        data_type='int16'
    )
    receiver.run()
