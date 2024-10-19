import torch

### Check ###
print(torch.cuda.is_available()) # IF all is GOOD it will return True 
print(torch.cuda.get_device_name(0)) # Will return your GPU name 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(device)



import cv2

def check_camera_and_show():
  """Checks if the default camera is working and shows a live feed.

  Args:
    None

  Returns:
    None
  """

  cap = cv2.VideoCapture(0)  # 0 usually refers to the default camera

  if not cap.isOpened():
    print("Error: Could not open camera.")
    return

  while True:
    ret, frame = cap.read()

    if not ret:
      print("Error: Could not read from camera.")
      break

    cv2.imshow('Live Feed', frame)

    if cv2.waitKey(1) == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

check_camera_and_show()