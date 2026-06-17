with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\classification\dataset.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix __init__ to handle image_size=None
old_init = '    def __init__(self, project_dir, split="train", transform=None, image_size=224):\n        self.project_dir = Path(project_dir)\n        self.split = split\n        self.image_size = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)'

new_init = '    def __init__(self, project_dir, split="train", transform=None, image_size=224):\n        self.project_dir = Path(project_dir)\n        self.split = split\n        self.image_size = image_size if (image_size is None or isinstance(image_size, (tuple, list))) else (image_size, image_size)'

if old_init in content:
    content = content.replace(old_init, new_init)
    print("1. Fixed __init__ for None image_size")

# Fix transform building to skip Resize when None
old_transform = '        if transform is None:\n            import torchvision.transforms as T\n            if split == "train":\n                self.transform = T.Compose([\n                    T.Resize(self.image_size),'

new_transform = '        if transform is None:\n            import torchvision.transforms as T\n            if self.image_size is None:\n                if split == "train":\n                    self.transform = T.Compose([\n                        T.RandomHorizontalFlip(p=0.5),\n                        T.RandomRotation(degrees=10),\n                        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),\n                        T.ToTensor(),\n                        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),\n                    ])\n                else:\n                    self.transform = T.Compose([\n                        T.ToTensor(),\n                        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),\n                    ])\n            elif split == "train":\n                self.transform = T.Compose([\n                    T.Resize(self.image_size),'

if old_transform in content:
    content = content.replace(old_transform, new_transform)
    print("2. Fixed transforms for full image mode")
else:
    print("ERROR: transform block not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\classification\dataset.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
