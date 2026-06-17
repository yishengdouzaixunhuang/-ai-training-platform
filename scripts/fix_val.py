with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\classification\trainer.py", "r", encoding="utf-8") as f:
    content = f.read()

# Val loader 1
old_v1 = '        self.val_loader = DataLoader(\n            val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False,\n            num_workers=2, pin_memory=True, prefetch_factor=2\n        )'
new_v1 = '        self.val_loader = DataLoader(\n            val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False,\n            num_workers=2, pin_memory=True, prefetch_factor=2,\n            collate_fn=collate\n        )'
if old_v1 in content:
    content = content.replace(old_v1, new_v1)
    print("Fixed val_loader 1")

# Val loader 2 (k-fold)
old_v2 = '            self.val_loader = DataLoader(\n                val_subset, batch_size=min(batch_size, len(val_subset)),\n                shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2\n            )'
new_v2 = '            self.val_loader = DataLoader(\n                val_subset, batch_size=min(batch_size, len(val_subset)),\n                shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2,\n                collate_fn=collate\n            )'
if old_v2 in content:
    content = content.replace(old_v2, new_v2)
    print("Fixed val_loader 2")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\classification\trainer.py", "w", encoding="utf-8") as f:
    f.write(content)
