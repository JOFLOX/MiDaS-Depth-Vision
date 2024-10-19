# Import Dep
import cv2
import torch
import time
import numpy as np



# Load the MiDaS model 

#model_type = "DPT_Large"     # MiDaS v3 - Large    (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"    # MiDaS v3 - Hybrid   (medium accuracy, medium inference speed)
model_type = "MiDaS_small"   # MiDaS v2.1 - Small  (lowest accuracy, highest inference speed) 

midas = torch.hub.load("intel-isl/MiDaS", model_type) # Download the model for the first time




# Set the model to GPU if availabe using CUDA (You must have a NVIDIA GPU)

### To use CUDA follow the next ### 
# 1- Update your Geforce to update your driver
# 2- install CUDA ToolKit (https://developer.nvidia.com/cuda-downloads) you can use the network installer 
# 3- Install PyTorch (https://pytorch.org/get-started/locally/)
### Note: it is OK if your CUDA 12.6 and only available CUDA 12.4 at PyTorch

### Check ###
#print(torch.cuda.is_available()) # IF all is GOOD it will return True 
#print(torch.cuda.get_device_name(0)) # Will return your GPU name 

### advice: restart your venv 



# Move the model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()     # Disabling training-specific operations like dropout and batch normalization



# Load transform to Resize and Normalize the Images 
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform



# Open the video capture from a webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    
    success, img = cap.read()
    start = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MiDaS is trained on RGB images 

    # Apply transforms
    input_batch = transform(img).to(device)

    # Prediction and resize to the original resolutoin 
    with torch.no_grad():   # Prevents the computation of gradients for the model's weights during the prediction
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode = "bicubic",
            align_corners = False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy() # Can't convert CUDA device type tensor to numpy. So, we use .cpu()

    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)


    end = time.time()
    fps = 1 / (end - start)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)


    cv2.putText(img, f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('IMAGE', img)
    cv2.imshow('DEPTH MAP', depth_map)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    

q

