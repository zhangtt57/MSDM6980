import subprocess

# 示例：调用7-Zip来解压文件
input_file_path = 'C:/Users/19433/Desktop/waveform/waveform.data.Z'
output_dir = 'C:/Users/19433/Desktop/waveform/'

try:
    # 运行7-Zip命令来解压缩文件
    subprocess.run(['7z', 'e', '-y', f'-o{output_dir}', input_file_path], check=True)
    print("文件解压缩完成")
except subprocess.CalledProcessError as e:
    print(f"解压缩失败: {e}")