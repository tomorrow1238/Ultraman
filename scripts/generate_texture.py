# common utils
import os
import argparse
import time

# pytorch3d
from pytorch3d.renderer import TexturesUV

# torch
import torch

from torchvision import transforms
import cv2
# numpy
import numpy as np

# image
from PIL import Image


# customized
import sys
sys.path.append(".")

from lib.mesh_helper import (
    init_mesh,
    apply_offsets_to_mesh,
    adjust_uv_map
)
from lib.render_helper import render
from lib.io_helper import (
    save_backproject_obj,
    save_args,
    save_viewpoints
)
from lib.vis_helper import (
    visualize_outputs, 
    visualize_principle_viewpoints, 
    visualize_refinement_viewpoints
)
from lib.diffusion_helper import (
    get_controlnet_depth,
    get_inpainting,
    apply_controlnet_depth,
    apply_inpainting_postprocess,
    get_sdxl_refiner,
    get_controlnet_depth_ipadapter,
    get_controlnet_depth_ipadapter_sdxl
)
from lib.projection_helper import (
    backproject_from_image,
    render_one_view_and_build_masks,
    select_viewpoint,
    build_similarity_texture_cache_for_all_views,
    get_canny,
    get_processed_canny,
    get_cleanup
)
from lib.camera_helper import init_viewpoints

from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline, 
    EulerDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionControlNetPipeline, 
    DDIMScheduler, 
    AutoencoderKL, 
    ControlNetModel,
    StableDiffusionXLControlNetPipeline
)
from models.IP_Adapter.ip_adapter.ip_adapter import IPAdapter, IPAdapterPlus, IPAdapterXL

# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()


def init_args():
    print("=> initializing input arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--origin_image_path", type=str, default="./origin_image.png")
    parser.add_argument("--obj_name", type=str, required=True)
    parser.add_argument("--obj_file", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--a_prompt", type=str, default="best quality, high quality, extremely detailed, good geometry")
    parser.add_argument("--n_prompt", type=str, default="deformed, extra digit, fewer digits, cropped, worst quality, low quality, smoke,paintings, cartoon, anime, sketches, ugly, blurry, Tan skin, dark skin, black skin, skin spots, skin blemishes, age spot, glans, disabled, distorted, bad anatomy, morbid, inconsistent skin, bad shoes.")
    parser.add_argument("--new_strength", type=float, default=1)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--num_inference_steps",type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--view_threshold", type=float, default=0.1)
    parser.add_argument("--viewpoint_mode", type=str, default="predefined", choices=["predefined", "hemisphere"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_unnormalized", action="store_true", help="save unnormalized mesh")

    parser.add_argument("--no_update", action="store_true", help="do NOT apply update")

    parser.add_argument("--add_view_to_prompt", action="store_true", help="add view information to the prompt")
    parser.add_argument("--post_process", action="store_true", help="post processing the texture")

    parser.add_argument("--smooth_mask", action="store_true", help="smooth the diffusion mask")

    # device parameters
    parser.add_argument("--device", type=str, choices=["person"], default="person")

    # camera parameters NOTE need careful tuning!!!
    parser.add_argument("--test_camera", action="store_true")
    parser.add_argument("--dist", type=float, default=1, 
        help="distance to the camera from the object")
    parser.add_argument("--elev", type=float, default=0,
        help="the angle between the vector from the object to the camera and the horizontal plane")
    parser.add_argument("--azim", type=float, default=180,
        help="the angle between the vector from the object to the camera and the vertical plane")

    args = parser.parse_args()

    if args.device == "person":
        setattr(args, "render_simple_factor", 4)
        setattr(args, "fragment_k", 1)
        setattr(args, "image_size", 1024)
        setattr(args, "uv_size", 1000)

    return args


if __name__ == "__main__":
    args = init_args()

    # save
    output_dir = os.path.join(
        args.output_dir, 
        "{}-{}-{}-{}".format(
            str(args.seed),
            args.viewpoint_mode[0],
            str(args.new_strength),
            str(args.view_threshold)
        ),
    )

    os.makedirs(output_dir, exist_ok=True)
    print("=> OUTPUT_DIR:", output_dir)

    # init resources
    # init mesh
    mesh, _, faces, aux, principle_directions, mesh_center, mesh_scale = init_mesh(os.path.join(args.input_dir, args.obj_file), DEVICE)

    #init oringin texture
    init_texture_path = os.path.join(args.input_dir, 'mesh_normalized.png')
    init_texture = Image.open(init_texture_path).convert("RGB").resize((args.uv_size, args.uv_size))

    #init input image
    origin_image = Image.open(args.origin_image_path)

    # HACK adjust UVs for multiple materials
    new_verts_uvs = aux.verts_uvs

    # update the mesh
    mesh.textures = TexturesUV(
        maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=new_verts_uvs[None, ...]
    )
    # back-projected faces
    exist_texture = torch.from_numpy(np.zeros([args.uv_size, args.uv_size]).astype(np.float32)).to(DEVICE)

    # initialize viewpoints for 8 directions
    dist_list = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
    elev_list = [15, 15, 15, 15, 15, 15, 15, 15, 90, -90]
    azim_list = [0, 45, 315, 90, 270, 135, 225, 180, 0, 0]
    sector_list = ['front', 'front right', 'front left', 'right', 'left', 'back right', 'back left', 'back', 'top', 'bottom']
    view_punishments = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # save args
    save_args(args, output_dir)

    # ------------------- LOAD MODEL ZONE BELOW ------------------------

    # base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    # image_encoder_path = "models/IP_Adapter/models/image_encoder"
    # ip_ckpt = "models/IP_Adapter/sdxl_models/ip-adapter_sdxl_vit-h.bin"
    # device = "cuda"

    # # load controlnet
    # controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"
    # controlnet = ControlNetModel.from_pretrained(controlnet_path, variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)

    # pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    #     base_model_path,
    #     controlnet=controlnet,
    #     use_safetensors=True,
    #     torch_dtype=torch.float16,
    #     add_watermarker=False,
    # ).to(device)

    # ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

    # ------------------- OPERATION ZONE BELOW ------------------------

    # 1. generate texture with RePaint 
    # NOTE no update / refinement

    generate_dir = os.path.join(output_dir, "generate")
    os.makedirs(generate_dir, exist_ok=True)

    update_dir = os.path.join(output_dir, "update")
    os.makedirs(update_dir, exist_ok=True)

    init_image_dir = os.path.join(generate_dir, "rendering")
    os.makedirs(init_image_dir, exist_ok=True)

    normal_map_dir = os.path.join(generate_dir, "normal")
    os.makedirs(normal_map_dir, exist_ok=True)

    mask_image_dir = os.path.join(generate_dir, "mask")
    os.makedirs(mask_image_dir, exist_ok=True)

    depth_map_dir = os.path.join(generate_dir, "depth")
    os.makedirs(depth_map_dir, exist_ok=True)

    similarity_map_dir = os.path.join(generate_dir, "similarity")
    os.makedirs(similarity_map_dir, exist_ok=True)

    inpainted_image_dir = os.path.join(generate_dir, "inpainted")
    os.makedirs(inpainted_image_dir, exist_ok=True)

    mesh_dir = os.path.join(generate_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)

    interm_dir = os.path.join(generate_dir, "intermediate")
    os.makedirs(interm_dir, exist_ok=True)

    canny_dir = os.path.join(generate_dir, "canny")
    os.makedirs(canny_dir, exist_ok = True)
    
    cleanup_dir = os.path.join(generate_dir, "clean")
    os.makedirs(cleanup_dir, exist_ok = True)


    # prepare viewpoints and cache
    NUM_PRINCIPLE = 10
    pre_dist_list = dist_list[:NUM_PRINCIPLE]
    pre_elev_list = elev_list[:NUM_PRINCIPLE]
    pre_azim_list = azim_list[:NUM_PRINCIPLE]
    pre_sector_list = sector_list[:NUM_PRINCIPLE]
    pre_view_punishments = view_punishments[:NUM_PRINCIPLE]

    pre_similarity_texture_cache = build_similarity_texture_cache_for_all_views(mesh, faces, new_verts_uvs,
        pre_dist_list, pre_elev_list, pre_azim_list,
        args.image_size, args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
        DEVICE
    )


    # start generation
    print("=> start generating texture...")
    start_time = time.time()
    for view_idx in range(NUM_PRINCIPLE):
        print("=> processing view {}...".format(view_idx))

        # sequentially pop the viewpoints
        dist, elev, azim, sector = pre_dist_list[view_idx], pre_elev_list[view_idx], pre_azim_list[view_idx], pre_sector_list[view_idx] 
        prompt = " the {} view of {}".format(sector, args.prompt) if args.add_view_to_prompt else args.prompt
        print("=> generating image for prompt: {}...".format(prompt))

        # 1.1. render and build masks
        (
            view_score,
            renderer, cameras, fragments,
            init_image, normal_map, depth_map, 
            init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor, 
            keep_mask_image, update_mask_image, generate_mask_image, all_mask_image,
            keep_mask_tensor, update_mask_tensor, generate_mask_tensor, all_mask_tensor, quad_mask_tensor,
        ) = render_one_view_and_build_masks(dist, elev, azim, 
            view_idx, view_idx, view_punishments, # => actual view idx and the sequence idx 
            pre_similarity_texture_cache, exist_texture,
            mesh, faces, new_verts_uvs,
            args.image_size, args.uv_size, args.fragment_k,
            init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir, output_dir,
            DEVICE, save_intermediate=True, smooth_mask=args.smooth_mask, view_threshold=args.view_threshold
        )

        # 1.2. generate missing region
        # NOTE first view still gets the mask for consistent ablations
        print("=> generate for view {}".format(view_idx))

        if view_idx == 0:
            generate_image = init_image.convert("RGBA")
            generate_image_before = init_image.convert("RGBA")
            generate_image_after = init_image.convert("RGBA")
            
        # When sometimes you can get the correct image in the top view, you can try this method
        # elif view_idx >= 8:
        #     (
        #         view_score1,
        #         renderer1, cameras1, fragments1,
        #         init_image1, normal_map1, depth_map_back, 
        #         init_images_tensor1, normal_maps_tensor1, depth_maps_tensor_back, similarity_tensor1, 
        #         keep_mask_image1, update_mask_image1, generate_mask_image1, all_mask_image1,
        #         keep_mask_tensor1, update_mask_tensor1, generate_mask_tensor1, all_mask_tensor1, quad_mask_tensor1,
        #     ) = render_one_view_and_build_masks(dist, 15, 180, 
        #         7, 7, view_punishments, # => actual view idx and the sequence idx 
        #         pre_similarity_texture_cache, exist_texture,
        #         mesh, faces, new_verts_uvs,
        #         args.image_size, args.uv_size, args.fragment_k,
        #         init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir, output_dir,
        #         DEVICE, save_intermediate=False, smooth_mask=args.smooth_mask, view_threshold=args.view_threshold
        #     )
        #     print("=> generating the image of back view of the human")
        #     prompt_back = "the back view of {}".format(args.prompt)
        #     back_view_image = get_controlnet_depth_ipadapter_sdxl(origin_image, depth_map_back, prompt_back, args.n_prompt,
        #     args.num_inference_steps, args.new_strength, args.guidance_scale, args.seed)

        #     generate_image = get_controlnet_depth_ipadapter_sdxl(back_view_image, depth_map, prompt, args.n_prompt, 
        #         args.num_inference_steps, args.new_strength, args.guidance_scale, args.seed)

        else:
            generate_image = get_controlnet_depth_ipadapter_sdxl(origin_image, depth_map, prompt, args.n_prompt, 
                args.num_inference_steps, args.new_strength, args.guidance_scale, args.seed)


        generate_image.save(os.path.join(inpainted_image_dir, "{}.png".format(view_idx)))

        # 1.2.2 back-project and create texture
        # NOTE projection mask = generate mask
        init_texture, project_mask_image, exist_texture = backproject_from_image(
            mesh, faces, new_verts_uvs, cameras, 
            generate_image.convert("RGB"), generate_mask_image, generate_mask_image, init_texture, exist_texture, 
            args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
            DEVICE
        )

        project_mask_image.save(os.path.join(mask_image_dir, "{}_project.png".format(view_idx)))

        # update the mesh
        mesh.textures = TexturesUV(
            maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
            faces_uvs=faces.textures_idx[None, ...],
            verts_uvs=new_verts_uvs[None, ...]
        )

        # 1.2.3. re: render 
        # NOTE only the rendered image is needed - masks should be re-used
        (
            view_score,
            renderer, cameras, fragments,
            init_image, *_,
        ) = render_one_view_and_build_masks(dist, elev, azim, 
            view_idx, view_idx, view_punishments, # => actual view idx and the sequence idx 
            pre_similarity_texture_cache, exist_texture,
            mesh, faces, new_verts_uvs,
            args.image_size, args.uv_size, args.fragment_k,
            init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir, output_dir,
            DEVICE, save_intermediate=False, smooth_mask=args.smooth_mask, view_threshold=args.view_threshold
        )

        # 1.3. update blurry region
        if not args.no_update and update_mask_tensor.sum() > 0 and update_mask_tensor.sum() / (all_mask_tensor.sum()) > 0.05:
            print("=> update {} pixels for view {}".format(update_mask_tensor.sum().int(), view_idx))
            diffused_image = generate_image

            diffused_image.save(os.path.join(inpainted_image_dir, "{}_update.png".format(view_idx)))

            # 1.3.2. back-project and create texture
            # NOTE projection mask = generate mask

            init_texture, project_mask_image, exist_texture = backproject_from_image(
                mesh, faces, new_verts_uvs, cameras, 
                diffused_image, update_mask_image, update_mask_image, init_texture, exist_texture, 
                args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
                DEVICE
            )
            
            # update the mesh
            mesh.textures = TexturesUV(
                maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
                faces_uvs=faces.textures_idx[None, ...],
                verts_uvs=new_verts_uvs[None, ...]
            )

        #get the intermediate results
        inter_images_tensor, *_ = render(mesh, renderer)
        inter_image = inter_images_tensor[0].cpu()
        inter_image = inter_image.permute(2, 0, 1)
        inter_image = transforms.ToPILImage()(inter_image).convert("RGB")
        inter_image.save(os.path.join(interm_dir, "{}.png".format(view_idx)))

        inter_images_tensor, *_ = render(mesh, renderer)
        inter_image = inter_images_tensor[0].cpu()
        inter_image = inter_image.permute(2, 0, 1)
        inter_image = transforms.ToPILImage()(inter_image).convert("RGB")
        inter_image.save(os.path.join(interm_dir, "{}.png".format(view_idx)))
        
        inter_image_path = os.path.join(interm_dir, "{}.png".format(view_idx))
        canny_path = os.path.join(canny_dir, "{}.png".format(view_idx))
        canny_processed_path = os.path.join(canny_dir, "{}_processed.png".format(view_idx))
        cleanup_path = os.path.join(cleanup_dir, "{}.png".format(view_idx))

        #seam smooth part
        if view_idx >= 5:
            old_mask_path = os.path.join(mask_image_dir, "{}_old.png".format(view_idx))
            new_mask_path = os.path.join(mask_image_dir, "{}_update.png".format(view_idx))
            old_mask = cv2.imread(old_mask_path, cv2.IMREAD_GRAYSCALE)
            new_mask = cv2.imread(new_mask_path, cv2.IMREAD_GRAYSCALE)
            get_canny(old_mask, new_mask, canny_path)

            get_processed_canny(canny_path, canny_processed_path)
            
            get_cleanup(inter_image_path, canny_processed_path, cleanup_dir, view_idx)
            
            clean_image = Image.open(cleanup_path)
            
            init_texture, project_mask_image, exist_texture = backproject_from_image(
                mesh, faces, new_verts_uvs, cameras, 
                clean_image, all_mask_image, all_mask_image, init_texture, exist_texture, 
                args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
                DEVICE
            )
            
            # update the mesh
            mesh.textures = TexturesUV(
                maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
                faces_uvs=faces.textures_idx[None, ...],
                verts_uvs=new_verts_uvs[None, ...]
            )

        # 1.4. save generated assets
        # save backprojected OBJ file
        save_backproject_obj(
            mesh_dir, "{}.obj".format(view_idx),
            mesh_scale * mesh.verts_packed() + mesh_center if args.use_unnormalized else mesh.verts_packed(),
            faces.verts_idx, new_verts_uvs, faces.textures_idx, init_texture, 
            DEVICE
        )

        # save texture mask
        exist_texture_image = exist_texture * 255. 
        exist_texture_image = Image.fromarray(exist_texture_image.cpu().numpy().astype(np.uint8)).convert("L")
        exist_texture_image.save(os.path.join(mesh_dir, "{}_texture_mask.png".format(view_idx)))

    print("=> total generate time: {} s".format(time.time() - start_time))

    # visualize viewpoints
    visualize_principle_viewpoints(output_dir, pre_dist_list, pre_elev_list, pre_azim_list)
