import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

celeba = "/home/teaching/Public/ddim/data/celeba/img_align_celeba"
generated = "/home/teaching/Public/ddim/cached_images_deep"

image_size = 299
batch_size = 32

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

class ImageFolderNoClass(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return None
        if self.transform:
            img = self.transform(img)
        return img

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return torch.stack(batch) if batch else torch.empty(0)

real_dataset = ImageFolderNoClass(celeba, transform)
real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, collate_fn=collate_skip_none)

fake_dataset = ImageFolderNoClass(generated, transform)
fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, collate_fn=collate_skip_none)

fid = FrechetInceptionDistance(normalize=True).to(device)

print("Computing FID features for CelebA images...")
for imgs in tqdm(real_loader):
    if imgs.numel() == 0:
        continue
    imgs = imgs.to(device)
    fid.update(imgs, real=True)

print("Computing FID features for generated images...")
for imgs in tqdm(fake_loader):
    if imgs.numel() == 0:
        continue
    imgs = imgs.to(device)
    fid.update(imgs, real=False)

print("Computing FID score...")
fid_score = fid.compute()
print(f"FID score: {fid_score.item():.4f}")
