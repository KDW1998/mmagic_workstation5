# python configs/edsr/SRData.py --source '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/학습데이터/실제촬영이미지/BKim_Thesis/leftImg8bit/train' --output '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/학습데이터/실제촬영이미지/SRData' --split val
import os
import argparse
from PIL import Image

def prepare_sisr_dataset(source_folder, output_root, split, scale=4):
    """
    Prepare dataset for MMagic SISR training.

    Args:
        source_folder (str): Path to folder containing high-resolution images.
        output_root (str): Base output folder.
        split (str): 'train', 'val', or 'test' to specify dataset split.
        scale (int): Downscaling factor for LR images (default: 4).
    """
    if split not in ['train', 'val', 'test']:
        raise ValueError("split must be one of: 'train', 'val', 'test'")

    split_dir = os.path.join(output_root, split)
    hr_dir = os.path.join(split_dir, 'HR')
    lr_dir = os.path.join(split_dir, 'LR')
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    images = sorted([f for f in os.listdir(source_folder)
                     if os.path.splitext(f)[1].lower() in exts])

    if not images:
        raise ValueError(f"No images found in {source_folder} with extensions {exts}")

    ann_file_name = f'meta_info_{split}.txt'
    ann_path = os.path.join(split_dir, ann_file_name)

    with open(ann_path, 'w') as ann_file:
        for fname in images:
            hr_path = os.path.join(source_folder, fname)
            hr_img = Image.open(hr_path)
            hr_img.save(os.path.join(hr_dir, fname))

            w, h = hr_img.size
            lr_size = (w // scale, h // scale)
            lr_img = hr_img.resize(lr_size, Image.BICUBIC)
            lr_img.save(os.path.join(lr_dir, fname))

            ann_file.write(f"{fname}\n")

    print("Dataset preparation complete.")
    print(f" - HR images saved in: {hr_dir}")
    print(f" - LR images saved in: {lr_dir}")
    print(f" - Annotation file saved at: {ann_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset for MMagic SISR training.')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to folder containing high-resolution images.')
    parser.add_argument('--output', type=str, required=True,
                        help='Base output folder.')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'],
                        help='Dataset split: train, val, or test.')
    parser.add_argument('--scale', type=int, default=4,
                        help='Downscaling factor for LR images (default: 4).')

    args = parser.parse_args()

    prepare_sisr_dataset(args.source, args.output, args.split, args.scale)
