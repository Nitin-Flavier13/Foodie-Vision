
from torchvision import transforms

train_transform1 = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # Random crop & resize
    transforms.RandomHorizontalFlip(p=0.5),  # Flip images
    transforms.RandomRotation(15),  # Rotate images
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Color adjustments
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # Rotation & shift
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Perspective distortion
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Apply blur
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform1 = transforms.Compose([
    transforms.Resize(size=(128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
