import os
import shutil
import random
from tqdm import tqdm


def split_dataset(original_dir,target_dir,train_ratio=0.8):
    classes = os.listdir(original_dir)
    os.makedirs(os.path.join(target_dir,"train"),exist_ok=True)
    os.makedirs(os.path.join(target_dir,"test"),exist_ok=True)

    for cls in tqdm(classes,desc="Splitting Dataset"):
        class_path = os.path.join(original_dir,cls)

        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        split_id = int(len(images) * train_ratio)
        train_img = images[:split_id]
        test_img = images[split_id:]

        os.makedirs(os.path.join(target_dir,"train",cls),exist_ok=True)
        os.makedirs(os.path.join(target_dir,"test",cls),exist_ok=True)

        for img in train_img:
            src = os.path.join(class_path,img)
            dst = os.path.join(target_dir,"train",cls,img)
            shutil.copy2(src, dst)

        for img in test_img:
            src = os.path.join(class_path,img)
            dst = os.path.join(target_dir,"test",cls,img)
            shutil.copy2(src, dst)


if __name__ == "__main__":
    original_dataset = "../data/doodle"
    target_dataset = "../data/doodle_split"
    split_dataset(original_dataset, target_dataset)