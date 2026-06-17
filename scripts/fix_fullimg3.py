with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\classification\trainer.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix predict() to handle None image_size
old_predict_resize = '            T.Resize((self._image_size, self._image_size)),'
new_predict_resize = '            T.Resize((self._image_size, self._image_size)) if self._image_size is not None else T.Lambda(lambda x: x),'

if old_predict_resize in content:
    content = content.replace(old_predict_resize, new_predict_resize)
    print("Fixed predict() for None image_size")
else:
    print("ERROR: not found")

# Fix _export_for_inference dummy shape
old_dummy = '        dummy = torch.randn(1, 3, self._image_size, self._image_size).to(self.device)'
new_dummy = '        img_sz = self._image_size if self._image_size is not None else 224\n        dummy = torch.randn(1, 3, img_sz, img_sz).to(self.device)'
if old_dummy in content:
    content = content.replace(old_dummy, new_dummy)
    print("Fixed export for None image_size")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\classification\trainer.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
