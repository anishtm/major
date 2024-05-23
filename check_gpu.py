import torch
import torchvision

# Check versions
print(f"Torch Version: {torch.__version__}")  # Should print 2.3.0+cu121
print(f"Torchvision Version: {torchvision.__version__}")  # Should print the updated version

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")

# Check device details
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"Device capability: {torch.cuda.get_device_capability(0)}")

print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")


# Validate NMS operation
from torchvision.ops import nms

boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 9, 9], [10, 10, 20, 20]], dtype=torch.float32).cuda()
scores = torch.tensor([0.9, 0.75, 0.8]).cuda()

keep = nms(boxes, scores, 0.5)
print(f"NMS output: {keep}")
