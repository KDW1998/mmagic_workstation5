# python configs/edsr/SRData_1024.py --source '/home/user/05. DIV2K/DIV2K_valid_HR' --output '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/DIV2K' --split val
import os
import argparse
from PIL import Image

def crop_with_overlap(img, crop_size, stride):
    w, h = img.size
    crops = []
    for top in range(0, h - crop_size + 1, stride):
        for left in range(0, w - crop_size + 1, stride):
            box = (left, top, left + crop_size, top + crop_size)
            crops.append(img.crop(box))
    return crops

def prepare_sisr_dataset(source_folder, output_root, split, scale=4, crop_size=1024, overlap=0.5):
    if split not in ['train', 'val', 'test']:
        raise ValueError("split must be one of: 'train', 'val', 'test'")

    stride = int(crop_size * (1 - overlap))
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
            w, h = hr_img.size

            crops = crop_with_overlap(hr_img, crop_size, stride)
            if not crops:
                print(f"Warning: {fname} is too small for cropping, skipping.")
                continue

            base_name = os.path.splitext(fname)[0]

            # Strip _leftImg8bit if present
            if base_name.endswith('_leftImg8bit'):
                base_name_clean = base_name[:-len('_leftImg8bit')]
            else:
                base_name_clean = base_name

            for idx, crop in enumerate(crops):
                crop_filename = f"{base_name_clean}_crop{idx}_leftImg8bit.png"
                crop.save(os.path.join(hr_dir, crop_filename))

                lr_size = (crop_size // scale, crop_size // scale)
                lr_img = crop.resize(lr_size, Image.BICUBIC)
                lr_img.save(os.path.join(lr_dir, crop_filename))

                ann_file.write(f"{crop_filename}\n")

    print("Dataset preparation complete.")
    print(f" - HR patches saved in: {hr_dir}")
    print(f" - LR patches saved in: {lr_dir}")
    print(f" - Annotation file saved at: {ann_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset for MMagic SISR training with cropping.')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to folder containing high-resolution images.')
    parser.add_argument('--output', type=str, required=True,
                        help='Base output folder.')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'],
                        help='Dataset split: train, val, or test.')
    parser.add_argument('--scale', type=int, default=4,
                        help='Downscaling factor for LR images (default: 4).')
    parser.add_argument('--crop_size', type=int, default=1024,
                        help='Crop size for HR patches (default: 1024).')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Fractional overlap between crops (default: 0.5).')

    args = parser.parse_args()

    prepare_sisr_dataset(args.source, args.output, args.split,
                         args.scale, args.crop_size, args.overlap)
