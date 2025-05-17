#python demo/mmagic_folder_inferencer_folder.py --model-name edsr --model-config 'work_dirs/edsr_x4c64b16_1xb16-300k_div2k/edsr_x4c64b16_1xb16-300k_div2k.py' --model-ckpt '/home/user/WindowsShare/05. Data/03. Checkpoints/hardnegative/SuperResolution/HNS1200andPS800/edsr_x4c64b16_1xb16-300k_div2k/iter_75000.pth' --img-dir '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/SRData_total/val/LR' --result-out-dir '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/SR_segmentation_train/leftImg8bit/val'
# Copyright (c) OpenMMLab. All rights reserved.
# isort: off
from argparse import ArgumentParser
from mmengine import DictAction
from mmagic.apis import MMagicInferencer
import os

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
        # Ensure the input directory exists
        assert os.path.isdir(args.img_dir), f"'{args.img_dir}' is not a valid directory."

        # Loop through each file in the directory
        for img_name in os.listdir(args.img_dir):
            img_path = os.path.join(args.img_dir, img_name)
            
            # Check if the file is an image (you can extend this to other formats if needed)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                output_path = os.path.join(args.result_out_dir, img_name)
                
                # Update the args with the current image and output paths
                user_defined['img'] = img_path
                user_defined['result_out_dir'] = output_path

                editor.infer(**user_defined)
    else:
        editor.infer(**user_defined)

if __name__ == '__main__':
    main()