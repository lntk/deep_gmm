# # #%%
# # import sampling
# # import torch

# # weights = torch.rand(50, 1, 4, 128, 128)
# # means = torch.rand(50, 1, 4, 128, 128)

# # samples = sampling.sequence_sample_bernoulli_mixtures(weights, means, channel_first=True)
# # print(samples[0].shape)
# # print(samples[0])

# #%%
# from util import image_utils

# image_utils.frame_files_to_video(frames_path="data/CDnet2014/baseline/highway/input/",
#                                  video_path="data/video.avi")

# print("Done")
# #%%
# # === TEST DATASET LOADER ===
# from dataset import VideoSequenceDataset
# from torchvision import transforms
# from transform import Rescale, ToTensor, Normalize

# video_dataset = VideoSequenceDataset(data_file="data/toy_train.txt",
#                                      path_to_data="data/toy",
#                                      transform=transforms.Compose([
#                                          Rescale(128),
#                                          Normalize(),
#                                          ToTensor()
#                                      ]))

# for i in range(len(video_dataset)):
#     frames, targets = video_dataset[i]
    
#     print(i, frames.size(), targets.size())
    
#     if i == 3:
#         break

#%%
from net.unet.model import UNet
import torch

model = UNet(n_channels=3, n_classes=2)

inputs = torch.Tensor(1, 3, 128, 128)
outputs = model(inputs)

print(outputs.shape)

#%%
