import os, shutil, random

random.seed(42)
src = "archive/flowers"   # <-- change this to whatever folder name `ls` showed you
dst = "data/images"

for cls in os.listdir(src):
    cls_src = os.path.join(src, cls)
    if not os.path.isdir(cls_src):
        continue
    cls_dst = os.path.join(dst, cls)
    os.makedirs(cls_dst, exist_ok=True)
    imgs = os.listdir(cls_src)
    random.shuffle(imgs)
    for f in imgs[:40]:
        shutil.copy(os.path.join(cls_src, f), os.path.join(cls_dst, f))

print("done")