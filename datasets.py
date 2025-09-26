import torch
import torchvision.transforms
    
def remove_background(img, bg_mask):
    # Remove background from img using bg_mask.
    return img * torch.logical_not(bg_mask)

class OtolithAgesDataset(torch.utils.data.Dataset):
    def __init__(self, img_data=None, age_data=None, bg_mask_data=None, augmentations=(False, False, False, False), bg_transform='orig', device='cuda'):
        self.device = device
        self.img_data = img_data
        self.age_data = age_data
        self.bg_mask_data = bg_mask_data
        self.bg_transform = bg_transform

        # Normalization using mean and variance of dataset.
        self.normalize = torchvision.transforms.Normalize([0.2638, 0.2333, 0.1550], [0.3758, 0.3352, 0.2359])

        self.color_augs = None
        self.flip_augs = None
        self.crop_augs = None
        self.rotate_augs = None

        # Define transformations for data augmentations.
        flip_augs, color_augs, crop_augs, rotate_augs = augmentations
        if flip_augs:
            self.flip_augs = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip()
            ])
        if color_augs:
            # Parameter values for ResNet.
            self.color_augs = torchvision.transforms.ColorJitter(brightness=(0.6, 1.1), contrast=(0.6, 1.1), saturation=(0.5, 1.25), hue=(-0.02, 0.02))
        if crop_augs:
            self.crop_augs = torchvision.transforms.RandomCrop(size=0)
        if rotate_augs:
            self.rotate_augs = torchvision.transforms.RandomRotation(degrees=10, fill=0)

    def __len__(self):
        return self.img_data.size(0)
    
    def __getitem__(self, index: int):
        img = self.img_data[index]
        h, w = img.size(1), img.size(2)

        # Color augmentations before background removal.
        if self.color_augs is not None:
            img = self.color_augs(img)

        # Only remove background if necessary.
        if self.bg_mask_data is not None:
            bg_mask = self.bg_mask_data[index]

            if self.bg_transform == 'rem':
                img = remove_background(img, bg_mask)

        # Crop, rotate, and flip augmentations after background removal.
        if self.crop_augs:
            self.crop_augs.size = (h - 32, w)
            img = self.crop_augs(img)
        
        if self.rotate_augs:
            img = self.rotate_augs(img)
        
        if self.flip_augs is not None:
            img = self.flip_augs(img)

        # Normalize images to zero mean and unit variance.
        img = self.normalize(img)

        return img, self.age_data[index]
    
    def vit_augs(self):
        # Parameter values for ViT.
        self.color_augs = torchvision.transforms.ColorJitter(brightness=(0.4, 1.3), contrast=(0.4, 1.3), saturation=(0.25, 1.5), hue=(-0.05, 0.05)) # ViT