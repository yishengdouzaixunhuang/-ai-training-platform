with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\classification\trainer.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add _pad_collate static method before train() method
pad_collate = '''
    @staticmethod
    def _pad_collate(batch):
        """Collate function that pads images to max H/W in the batch."""
        import torch.nn.functional as F
        images, labels = zip(*batch)
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        padded = []
        for img in images:
            if img.shape[1] < max_h or img.shape[2] < max_w:
                img = F.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1]))
            padded.append(img)
        return torch.stack(padded), torch.tensor(labels)
'''

train_marker = '    def train(self, epochs=50, batch_size=32, image_size=224,'
if train_marker in content and 'def _pad_collate' not in content:
    content = content.replace(train_marker, pad_collate + '\n' + train_marker)
    print("1. Added _pad_collate method")

# 2. Add collate_fn to DataLoader calls when image_size is None
# First DataLoader (normal training, ~L349)
old_dl1 = '        self.train_loader = DataLoader(\n            train_ds, batch_size=batch_size, shuffle=True,\n            num_workers=min(4, multiprocessing.cpu_count() or 4),\n            pin_memory=True, prefetch_factor=2, drop_last=drop\n        )'
new_dl1 = '        collate = self._pad_collate if image_size is None else None\n        self.train_loader = DataLoader(\n            train_ds, batch_size=batch_size, shuffle=True,\n            num_workers=min(4, multiprocessing.cpu_count() or 4),\n            pin_memory=True, prefetch_factor=2, drop_last=drop,\n            collate_fn=collate\n        )'

if old_dl1 in content:
    content = content.replace(old_dl1, new_dl1)
    print("2. Added collate_fn to normal train_loader")
else:
    print("ERROR: dl1 not found")

# Second DataLoader (k-fold, ~L491)
old_dl2 = '            self.train_loader = DataLoader(\n                train_subset, batch_size=batch_size, shuffle=True,\n                num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=drop\n            )'
new_dl2 = '            collate = self._pad_collate if image_size is None else None\n            self.train_loader = DataLoader(\n                train_subset, batch_size=batch_size, shuffle=True,\n                num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=drop,\n                collate_fn=collate\n            )'

if old_dl2 in content:
    content = content.replace(old_dl2, new_dl2)
    print("3. Added collate_fn to k-fold train_loader")
else:
    print("ERROR: dl2 not found")

# Also val loaders need collate
old_val1 = '        self.val_loader = DataLoader(\n            val_ds, batch_size=min(batch_size, len(val_ds)),\n            shuffle=False, num_workers=2, pin_memory=True, drop_last=False\n        )'
new_val1 = '        self.val_loader = DataLoader(\n            val_ds, batch_size=min(batch_size, len(val_ds)),\n            shuffle=False, num_workers=2, pin_memory=True, drop_last=False,\n            collate_fn=collate\n        )'

if old_val1 in content:
    content = content.replace(old_val1, new_val1)
    print("4. Added collate_fn to normal val_loader")
else:
    print("ERROR: val1 not found")

old_val2 = '            self.val_loader = DataLoader(\n                val_subset, batch_size=min(batch_size, len(val_subset)),\n                shuffle=False, num_workers=2, pin_memory=True, drop_last=False\n            )'
new_val2 = '            self.val_loader = DataLoader(\n                val_subset, batch_size=min(batch_size, len(val_subset)),\n                shuffle=False, num_workers=2, pin_memory=True, drop_last=False,\n                collate_fn=collate\n            )'

if old_val2 in content:
    content = content.replace(old_val2, new_val2)
    print("5. Added collate_fn to k-fold val_loader")
else:
    print("ERROR: val2 not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\classification\trainer.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
