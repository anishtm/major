import torch
import torchvision
from torchvision.ops import nms

boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 9, 9], [10, 10, 20, 20]], dtype=torch.float32).cuda()
scores = torch.tensor([0.9, 0.75, 0.8]).cuda()

keep = nms(boxes, scores, 0.5)
print(keep)
