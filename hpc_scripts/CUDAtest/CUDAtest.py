import torch

print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device:', torch.cuda.get_device_name(0))
    x = torch.rand(1024, 1024).cuda()
    y = torch.rand(1024, 1024).cuda()
    z = torch.mm(x, y)
    print('Sample result:', z[0][0].item())
else:
    print('CUDA not available.')
