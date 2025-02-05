
# MiDaS-based Depth Estimation 

This project demonstrates the use of the MiDaS (Multiple Depth Images from Single Image) model for depth estimation from a single image. MiDaS is a powerful deep learning model that can accurately estimate depth information, enabling various applications in computer vision, robotics, and autonomous driving.

## Demo

![alt text](https://github.com/JOFLOX/MiDaS-Depth-Vision-/blob/main/Demo/Screenshot%202024-10-19%20231430.jpg?raw=true) 

![alt text](https://github.com/JOFLOX/MiDaS-Depth-Vision-/blob/main/Demo/Screenshot%202024-10-19%20231804.jpg?raw=true)




## MiDaS Models 

model_type = "DPT_Large"    
MiDaS v3 - Large    (highest accuracy, slowest inference speed)

model_type = "DPT_Hybrid"    
MiDaS v3 - Hybrid   (medium accuracy, medium inference speed)

model_type = "MiDaS_small"   
MiDaS v2.1 - Small  (lowest accuracy, highest inference speed) 

## To use CUDA follow the next ### 
#### 1- Update your Geforce to update your driver
#### 2- install CUDA ToolKit (https://developer.nvidia.com/cuda-downloads) you can use the network installer 
#### 3- Install PyTorch (https://pytorch.org/get-started/locally/)
##### Note: it is OK if your CUDA 12.6 and only available CUDA 12.4 at PyTorch

### Check ###
##### print(torch.cuda.is_available()) # IF all is GOOD it will return True 
##### print(torch.cuda.get_device_name(0)) # Will return your GPU name 

#### advice: restart your venv 
### Documentation

[Pytorch-MIDAS](https://pytorch.org/hub/intelisl_midas_v2/#:~:text=By%20Intel%20ISL,depth%20from%20a%20single%20image.)


### 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/JOFLOX)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/youssef-sayed-joe)


