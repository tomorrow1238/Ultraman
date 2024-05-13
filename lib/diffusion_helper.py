import torch

import cv2
import numpy as np

from PIL import Image
from torchvision import transforms

# Stable Diffusion 2
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

# customized
import sys
sys.path.append(".")

from models.ControlNet.gradio_depth2image import init_model, process
from models.IP_Adapter.ip_adapter import IPAdapter, IPAdapterPlus, IPAdapterXL

def get_controlnet_depth_ipadapter(init_image, depth_map, prompt, negative_prompt, num_inference_steps=50, strength=0.1, guidance_scale=7.5, seed=42):
    base_model_path = "runwayml/stable-diffusion-v1-5"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "models/IP_Adapter/models/image_encoder/"
    ip_ckpt = "models/IP_Adapter/models/ip-adapter_sd15.bin"
    device = "cuda"

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    # load controlnet
    controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
    controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
    # load SD pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

    depth_map = depth_map.resize((1408, 1408))
    images = ip_model.generate(pil_image=init_image, prompt = prompt, negative_prompt = negative_prompt, image=depth_map, 
                                controlnet_conditioning_scale=0.7, num_samples=1, num_inference_steps = num_inference_steps, strength = strength, guidance_scale = guidance_scale, seed=seed)

    return images[0].resize((1024, 1024))


def get_controlnet_depth_ipadapter_sdxl(init_image, depth_map, prompt, negative_prompt, num_inference_steps=50, strength=0.1, guidance_scale=7.5, seed=42):
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    image_encoder_path = "models/IP_Adapter/models/image_encoder"
    ip_ckpt = "models/IP_Adapter/sdxl_models/ip-adapter_sdxl_vit-h.bin"
    device = "cuda"

    # load controlnet
    controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"
    controlnet = ControlNetModel.from_pretrained(controlnet_path, variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        use_safetensors=True,
        torch_dtype=torch.float16,
        add_watermarker=False,
    ).to(device)

    ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

    depth_map = depth_map.resize((1408, 1408))
    images = ip_model.generate(pil_image=init_image, prompt = prompt, negative_prompt = negative_prompt, image=depth_map, 
                                controlnet_conditioning_scale=0.7, num_samples=1, num_inference_steps = num_inference_steps, strength = strength, guidance_scale = guidance_scale, seed=seed)

    return images[0].resize((1024, 1024))

def get_controlnet_depth():
    print("=> initializing ControlNet Depth...")
    model, ddim_sampler = init_model()

    return model, ddim_sampler


def get_inpainting(device):
    print("=> initializing Inpainting...")

    model = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    ).to(device)

    return model

def get_text2image(device):
    print("=> initializing Inpainting...")

    model_id = "stabilityai/stable-diffusion-2"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    model = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16).to(device)

    return model


@torch.no_grad()
def apply_controlnet_depth(model, ddim_sampler, 
    init_image, prompt, strength, ddim_steps,
    generate_mask_image, keep_mask_image, depth_map_np, 
    a_prompt, n_prompt, guidance_scale, seed, eta, num_samples,
    device, blend=0, save_memory=False):
    """
        Use Stable Diffusion 2 to generate image

        Arguments:
            args: input arguments
            model: Stable Diffusion 2 model
            init_image_tensor: input image, torch.FloatTensor of shape (1, H, W, 3)
            mask_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W, 1)
            depth_map_np: depth map of the input image, torch.FloatTensor of shape (1, H, W)
    """

    print("=> generating ControlNet Depth RePaint image...")
    # import pdb

    # Stable Diffusion 2 receives PIL.Image
    # NOTE Stable Diffusion 2 returns a PIL.Image object
    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is
    diffused_image_np = process(
        model, ddim_sampler,
        np.array(init_image), prompt, a_prompt, n_prompt, num_samples,
        ddim_steps, guidance_scale, seed, eta, 
        strength=strength, detected_map=depth_map_np, unknown_mask=np.array(generate_mask_image), save_memory=save_memory
    )[0]

    init_image = init_image.convert("RGB")
    diffused_image = Image.fromarray(diffused_image_np).convert("RGB")
    # pdb.set_trace()

    if blend > 0 and transforms.ToTensor()(keep_mask_image).sum() > 0:
        print("=> blending the generated region...")
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        keep_image_np = np.array(init_image).astype(np.uint8)
        keep_image_np_dilate = cv2.dilate(keep_image_np, kernel, iterations=1)

        keep_mask_np = np.array(keep_mask_image).astype(np.uint8)
        keep_mask_np_dilate = cv2.dilate(keep_mask_np, kernel, iterations=1)

        generate_image_np = np.array(diffused_image).astype(np.uint8)

        overlap_mask_np = np.array(generate_mask_image).astype(np.uint8)
        overlap_mask_np *= keep_mask_np_dilate
        print("=> blending {} pixels...".format(np.sum(overlap_mask_np)))

        overlap_keep = keep_image_np_dilate[overlap_mask_np == 1]
        overlap_generate = generate_image_np[overlap_mask_np == 1]

        overlap_np = overlap_keep * blend + overlap_generate * (1 - blend)

        generate_image_np[overlap_mask_np == 1] = overlap_np

        diffused_image = Image.fromarray(generate_image_np.astype(np.uint8)).convert("RGB")

    init_image_masked = init_image
    diffused_image_masked = diffused_image

    return diffused_image, init_image_masked, diffused_image_masked


@torch.no_grad()
def apply_inpainting(model, 
    init_image, mask_image_tensor, prompt, height, width, device):
    """
        Use Stable Diffusion 2 to generate image

        Arguments:
            args: input arguments
            model: Stable Diffusion 2 model
            init_image_tensor: input image, torch.FloatTensor of shape (1, H, W, 3)
            mask_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W, 1)
            depth_map_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W)
    """

    print("=> generating Inpainting image...")

    mask_image = mask_image_tensor[0].cpu()
    mask_image = mask_image.permute(2, 0, 1)
    mask_image = transforms.ToPILImage()(mask_image).convert("L")

    # NOTE Stable Diffusion 2 returns a PIL.Image object
    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is
    diffused_image = model(
        prompt=prompt, 
        image=init_image.resize((512, 512)), 
        mask_image=mask_image.resize((512, 512)), 
        height=512, 
        width=512
    ).images[0].resize((height, width))

    return diffused_image


@torch.no_grad()
def apply_inpainting_postprocess(model, 
    init_image, mask_image_tensor, prompt, height, width, device):
    """
        Use Stable Diffusion 2 to generate image

        Arguments:
            args: input arguments
            model: Stable Diffusion 2 model
            init_image_tensor: input image, torch.FloatTensor of shape (1, H, W, 3)
            mask_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W, 1)
            depth_map_tensor: depth map of the input image, torch.FloatTensor of shape (1, H, W)
    """

    print("=> generating Inpainting image...")

    mask_image = mask_image_tensor[0].cpu()
    mask_image = mask_image.permute(2, 0, 1)
    mask_image = transforms.ToPILImage()(mask_image).convert("L")

    # NOTE Stable Diffusion 2 returns a PIL.Image object
    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is
    diffused_image = model(
        prompt=prompt, 
        image=init_image.resize((512, 512)), 
        mask_image=mask_image.resize((512, 512)), 
        height=512, 
        width=512
    ).images[0].resize((height, width))

    diffused_image_tensor = torch.from_numpy(np.array(diffused_image)).to(device)

    init_images_tensor = torch.from_numpy(np.array(init_image)).to(device)
    
    init_images_tensor = diffused_image_tensor * mask_image_tensor[0] + init_images_tensor * (1 - mask_image_tensor[0])
    init_image = Image.fromarray(init_images_tensor.cpu().numpy().astype(np.uint8)).convert("RGB")

    return init_image

