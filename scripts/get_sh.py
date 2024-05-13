import argparse
import os

def get_sh(input_dir, output_dir,origin_image_path, obj_name, obj_file, prompt_file_path, output_sh_path):
    with open(prompt_file_path, 'r') as file:
        prompt = file.readline().strip()

    content = f'''CUDA_VISIBLE_DEVICES=0 python scripts/generate_texture.py \\
                    --input_dir {input_dir} \\
                    --output_dir {output_dir} \\
                    --origin_image_path {origin_image_path} \\
                    --obj_name {obj_name} \\
                    --obj_file {obj_file} \\
                    --prompt "{prompt}" \\
                    --add_view_to_prompt \\
                    --ddim_steps 50 \\
                    --new_strength 1 \\
                    --view_threshold 0.1 \\
                    --dist 0.7 \\
                    --viewpoint_mode predefined \\
                    --seed 42 \\
                    --post_process \\
                    --device person \\
    '''

    with open(output_sh_path, 'w') as file:
        file.write(content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="input path of obj")
    parser.add_argument("--output_dir", help="name of output folder")
    parser.add_argument("--origin_image_path", help="origin image path")
    parser.add_argument("--obj_name", help="obj name")
    parser.add_argument("--obj_file", help="obj file")
    parser.add_argument("--prompt_file_path", help="prompt file path")
    parser.add_argument("--output_sh_path", help="output path of sh file")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    origin_image_path = args.origin_image_path
    obj_name = args.obj_name
    obj_file = args.obj_file
    prompt_file_path = args.prompt_file_path
    output_sh_path = os.path.join(args.output_sh_path)

    get_sh(input_dir, output_dir, origin_image_path, obj_name, obj_file, prompt_file_path, output_sh_path)
