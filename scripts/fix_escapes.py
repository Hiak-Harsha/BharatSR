import pathlib

src = pathlib.Path("frontend/src")
fixed = []
for f in src.rglob("*.ts*"):
    content = f.read_text(encoding="utf-8")
    if '\\"' in content:
        new_content = content.replace('\\"', '"')
        f.write_text(new_content, encoding="utf-8")
        fixed.append(str(f))

print(f"Fixed {len(fixed)} files:")
for name in fixed:
    print(" -", name)
