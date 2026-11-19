import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
print("PyTorch 编译使用的 CUDA 版本:", torch.version.cuda)


print("可用 GPU 数量:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("当前显卡名称:", torch.cuda.get_device_name(0))

print("cuDNN 是否可用:", torch.backends.cudnn.is_available())
print("cuDNN 版本:", torch.backends.cudnn.version())

if torch.cuda.is_available():
    print("CUDA 算力:", torch.cuda.get_device_capability(0))
    print("总显存(GB):", torch.cuda.get_device_properties(0).total_memory / 1024**3)