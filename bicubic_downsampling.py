import os
import PIL.Image as Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.nn.functional import interpolate


img_dir = "dataset/train_hr_png/"
save_dir = "dataset/train_lr_png/"
os.makedirs(save_dir, exist_ok=True)
# print(os.listdir(img_dir)[:])
for i, f in enumerate(os.listdir(img_dir)):
    print(f'{i}/{len(os.listdir(img_dir))}')
    path = os.path.join(img_dir, f)

    image = Image.open(path)
    
    if image.mode == "L":
      image = image.convert('RGB')
    
    
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    
    image = transform(image).float()/255.0
    image = image.unsqueeze(0)

    image_lr = interpolate(image, size=160, mode='bicubic').clamp(min=0, max=255)
    out_path = os.path.join(save_dir, f)
    save_image(image_lr, out_path)