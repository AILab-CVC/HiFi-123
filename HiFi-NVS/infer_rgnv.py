from diffusers import StableDiffusionUpscalePipeline
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
from ldm.util_new import add_bg, load_image, resize_image, mat_image
import torchvision
from diffusers import DDIMScheduler
import cv2
from attention.diffuser_utils import RGNV
from attention.attention_utils import regiter_attention_editor_diffusers
from attention.attention import MutualSelfAttentionControl
from torchvision.utils import save_image
from pytorch_lightning import seed_everything
from midas.api import MiDaSInference 
import PIL
import argparse

def main(model_path,upscaler_path,stop_step,seed,target_prompt,source_image_path,target_image_path,bg_path,attn_step, attn_layer,out_path,device):

    os.makedirs(out_path,exist_ok=True)
    save_name = target_image_path.split('/')[-1].split('.')[0]
    seed_everything(seed)
    seed_up = torch.manual_seed(42)
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model = RGNV.from_pretrained(model_path, scheduler=scheduler, cross_attention_kwargs={"scale": 0.5}).to(device)
    upscaler = StableDiffusionUpscalePipeline.from_pretrained(upscaler_path, torch_dtype=torch.float16)
    upscaler = upscaler.to(device)
    upscaler.set_use_memory_efficient_attention_xformers(True)

    # source_image = load_image(source_image_path, device)
    source_image = add_bg(source_image_path,bg_path, device)
    # source_image,_,_ = mat_image(source_image_path, device)
    # target_image = load_image(target_image_path, device)
    target_image, mask_small, mask = mat_image(target_image_path,device)
    imgs = torch.cat([source_image,target_image],dim=0)

    depth_model = MiDaSInference(model_type="dpt_hybrid").to(device)
    cc = depth_model(imgs)
    cc[-1] = cc[-1]*mask
    depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],keepdim=True)
    cc = 2. * (cc - depth_min) / (depth_max - depth_min + 0.001) - 1.
    cc = torch.nn.functional.interpolate(
        cc,
        size=(64,64),
        mode="bicubic",
        align_corners=False,
    )

    source_prompt = ""
    prompts = [source_prompt, target_prompt] # construct batch/pair input

    # 1 4 64 64
    start_code, latents_list = model.invert(source_image,
                                            source_prompt,
                                            cc[0].unsqueeze(0),
                                            guidance_scale=9.,
                                            num_inference_steps=50,
                                            return_intermediates=True
                                            )
    start_code = start_code.expand(len(prompts), -1, -1, -1) #2 4 64 64, construct batch/pair input

    xt, _ = model.invert(target_image,
                        source_prompt,
                        cc[-1].unsqueeze(0),
                        guidance_scale=9.,
                        num_inference_steps=50,
                        return_intermediates=True,
                        stop = stop_step
                        )

    editor = MutualSelfAttentionControl(attn_step, attn_layer)
    regiter_attention_editor_diffusers(model, editor)


    out_image = model(prompts,cc, latents=start_code, guidance_scale=9., xt = xt,t_start = 50-stop_step,mask = mask_small)[-1:]
    # save_image(out_image, os.path.join(out_path,target_image_path.split('/')[-1].split('.')[0]+'_base.png'))
    save_image(mask*out_image+(1.-mask), os.path.join(out_path,save_name+'_base.png'), normalize=True, value_range=(0, 1))

    up_image = upscaler(prompt=target_prompt, image= F.interpolate(out_image, size=256, mode='bilinear', align_corners=False), generator=seed_up, noise_level=50).images[0]
    # up_image.save(os.path.join(out_path,target_image_path.split('/')[-1].split('.')[0]+'_upsample.png'))
    torch_image = transforms.ToTensor()(up_image).cuda()
    mask = F.interpolate(mask, size=(1024, 1024), mode='nearest')
    masked_image = torch_image * mask+(1.-mask)
    result_image = transforms.ToPILImage()(masked_image[0])
    result_image.save(os.path.join(out_path,save_name+'_upsample.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str, help="Device to be used.")
    parser.add_argument("--model_path", default="./stable-diffusion-2-depth", type=str, help="Path to the SD-depth model.")
    parser.add_argument("--upscaler_path", default="./stable-diffusion-x4-upscaler", type=str, help="Path to the SD upscaler model.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--target_prompt", default="Side view of a parrot.", type=str, help="Target prompt.")
    parser.add_argument("--stop_step", default=30, type=int, help="Stop step of DDIM inversion.")
    parser.add_argument("--source_image_path", default='./load/parrot.png', type=str, help="Path to the source image.")
    parser.add_argument("--target_image_path", default='./load/parrot_coarse.png', type=str, help="Path to the target image.")
    parser.add_argument("--bg_path", default='./load/bg2.png', type=str, help="Path to the background image.")
    parser.add_argument("--attn_step", default=4, type=int, help="attention injection step")
    parser.add_argument("--attn_layer", default=12, type=int, help="attention injection layer")
    parser.add_argument("--out_path", default='./output', type=str, help="Path to the output directory.")
    args = parser.parse_args()
    main(args.model_path,args.upscaler_path,args.stop_step,args.seed,args.target_prompt,args.source_image_path,args.target_image_path,args.bg_path, args.attn_step, args.attn_layer,args.out_path,args.device)

