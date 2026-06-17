with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\classification\trainer.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find where DataLoader is created
idx = content.find('DataLoader(')
if idx > 0:
    # Find the line
    start = content.rfind('\n', 0, idx)
    end = content.find('\n', idx)
    print("DataLoader line:", content[start:end].strip()[:120])
    print("---")
    # Find all DataLoader usages
    import re
    for m in re.finditer(r'DataLoader\(', content):
        line_start = content.rfind('\n', 0, m.start()) + 1
        line_end = content.find('\n', m.end())
        print(content[line_start:line_end].strip()[:150])
