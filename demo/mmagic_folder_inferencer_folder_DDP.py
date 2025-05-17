#python demo/mmagic_folder_inferencer_folder.py --model-name edsr --model-config configs/edsr/edsr_x4c64b16_1xb16-300k_div2k.py --model-ckpt configs/edsr/edsr_x4c64b16_1x16_300k_div2k_20200608-3c2af8a3.pth --img-dir 'raw' --result-out-dir 'raw/X4'

#python demo/mmagic_folder_inferencer폴더.py --model-name edsr --model-config configs/edsr/edsr_x4c64b16_1xb16-300k_div2k.py --model-ckpt configs/edsr/edsr_x4c64b16_1x16_300k_div2k_20200608-3c2af8a3.pth --img-dir '/home/deogwonkang/WindowsShare/05. Data/01. Under_Process/024. Korea Electricity/2024.03.27 초해상화 필요 이미지' --result-out-dir '/home/deogwonkang/crack_gauge'
# Copyright (c) OpenMMLab. All rights reserved.
# isort: off
from argparse import ArgumentParser
from mmengine import DictAction
from mmagic.apis import MMagicInferencer
import os
import multiprocessing
import torch

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--img', type=str, default=None, help='Input image file.')
    parser.add_argument(
        '--video', type=str, default=None, help='Input video file.')
    parser.add_argument('--img-dir', type=str, default=None, help='Input directory containing images to be processed.')
    parser.add_argument(
        '--label',
        type=int,
        default=None,
        help='Input label for conditional models.')
    parser.add_argument(
        '--trimap', type=str, default=None, help='Input for matting models.')
    parser.add_argument(
        '--mask',
        type=str,
        default=None,
        help='path to input mask file for inpainting models')
    parser.add_argument(
        '--text',
        type=str,
        default='',
        help='text input for text2image models')
    parser.add_argument(
        '--result-out-dir',
        type=str,
        default=None,
        help='Output img or video path.')
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Pretrained mmagic algorithm')
    parser.add_argument(
        '--model-setting',
        type=int,
        default=None,
        help='Pretrained mmagic algorithm setting')
    parser.add_argument(
        '--model-config',
        type=str,
        default=None,
        help='Path to the custom config file of the selected mmagic model.')
    parser.add_argument(
        '--model-ckpt',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected det model.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device used for inference.')
    parser.add_argument(
        '--extra-parameters',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for different model')
    parser.add_argument(
        '--seed',
        type=int,
        default=2022,
        help='The random seed used in inference.')
    # print supported tasks and models
    parser.add_argument(
        '--print-supported-models',
        action='store_true',
        help='print all supported models for inference.')
    parser.add_argument(
        '--print-supported-tasks',
        action='store_true',
        help='print all supported tasks for inference.')
    parser.add_argument(
        '--print-task-supported-models',
        type=str,
        default=None,
        help='print all supported models for one task')

    args, unknown = parser.parse_known_args()

    return args, unknown


def process_image(args_tuple):
    """
    Function to be executed in parallel for each image.
    args_tuple contains all necessary arguments for processing a single image.
    """
    img_path, output_dir, user_defined = args_tuple
    img_name = os.path.basename(img_path)
    output_path = os.path.join(output_dir, img_name)
    
    # Ensure the MMagicInferencer uses the correct GPU for this process
    local_args = user_defined.copy()
    local_args['img'] = img_path
    local_args['result_out_dir'] = output_path
    
    editor = MMagicInferencer(**local_args)
    editor.infer(**local_args)

def main():
    args, unknown = parse_args()
    assert len(unknown) % 2 == 0, (
        'User defined arguments must be passed in pair, but receive '
        f'{len(unknown)} arguments.')

    user_defined = {}
    for idx in range(len(unknown) // 2):
        key, val = unknown[idx * 2], unknown[idx * 2 + 1]
        assert key.startswith('--'), (
            'Key of user define arguments must be start with \'--\', but '
            f'receive \'{key}\'.')

        key = key.replace('-', '_')
        val = int(val) if val.isdigit() else val
        user_defined[key[2:]] = val

    user_defined.update(vars(args))

    if args.print_supported_models:
        inference_supported_models = \
            MMagicInferencer.get_inference_supported_models()
        print('all supported models:')
        print(inference_supported_models)
        return

    if args.print_supported_tasks:
        supported_tasks = MMagicInferencer.get_inference_supported_tasks()
        print('all supported tasks:')
        print(supported_tasks)
        return

    if args.print_task_supported_models:
        task_supported_models = \
            MMagicInferencer.get_task_supported_models(
                args.print_task_supported_models)
        print('translation models:')
        print(task_supported_models)
        return
    
    editor = MMagicInferencer(**vars(args))
    # Check if img-dir is provided and process all images in that directory
    if args.img_dir:
        assert os.path.isdir(args.img_dir), f"'{args.img_dir}' is not a valid directory."
        
        # Prepare a list of arguments for each image
        img_paths = [os.path.join(args.img_dir, img_name) for img_name in os.listdir(args.img_dir)
                     if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        process_args = [(img_path, args.result_out_dir, user_defined) for img_path in img_paths]

        # Determine the number of processes based on the number of available GPUs
        num_gpus = torch.cuda.device_count()  # Assuming PyTorch, adjust for your framework
        pool = multiprocessing.Pool(processes=num_gpus)

        # Start processing images in parallel
        pool.map(process_image, process_args)
        
    else:
        # Process single image as before
        editor = MMagicInferencer(**vars(args))
        editor.infer(**user_defined)

if __name__ == '__main__':
    main()