import torch
print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])

