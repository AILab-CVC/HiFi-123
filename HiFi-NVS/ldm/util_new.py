import importlib
import math
import torchvision
import torch
from torch import autocast
from torch import optim
import numpy as np
import torch.nn.functional as F
from torchvision.io import read_image
from lovely_numpy import lo
from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import time
import cv2
from ldm.high import HiInterface
import PIL
from torchvision import transforms
from contextlib import nullcontext

def add_bg(image_path,bg_path,device):
    carvekit = create_carvekit_interface()
    image = Image.open(image_path).convert('RGB').resize((512,512))
    image_without_background = carvekit([image])[0]
    image_without_background = np.array(image_without_background)
    est_seg = image_without_background > 127
    image = np.array(image)
    foreground = est_seg[:, : , -1].astype(np.float32)
    foreground = np.expand_dims(foreground, axis=-1)

    bg = cv2.cvtColor(
            cv2.imread(bg_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
    bg = (
        cv2.resize(
            bg, (512, 512), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
    )[:,:,:3]
    image_bg = image*foreground + bg*(1-foreground)
    image_bg = transforms.ToTensor()(image_bg/255.)*2-1
    image_bg = image_bg.unsqueeze(0).to(device)
    return image_bg

def load_image(image_path, device):
    totensor = transforms.ToTensor()
    image = totensor(Image.open(image_path))
    image = image[:3].unsqueeze_(0).float() *2.-1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image

def resize_image(image,size,mode='bilinear'):
    image = F.interpolate(image, (size,size),mode)
    return image

def mat_image(image_path,device):
    carvekit = create_carvekit_interface()
    image = Image.open(image_path).convert('RGB').resize((512,512))
    image_without_background = carvekit([image])[0]
    image_without_background = np.array(image_without_background)
    est_seg = image_without_background > 127
    image = np.array(image)
    foreground = est_seg[:, : , -1].astype(np.bool_)
    # select a pixel in reference image
    image[~foreground] = [126, 129, 127]
    # image[~foreground] = [97,123,48] #[126, 129, 127]

    fg = torch.from_numpy(foreground).float().unsqueeze(0).unsqueeze(0).to(device)
    fg_big = fg.clone()
    fg = torch.nn.functional.interpolate(
    fg,
    size=(64,64),
    mode="nearest",
)
    image = transforms.ToTensor()(image)*2-1
    image = image.unsqueeze(0).to(device)
    return image,fg,fg_big

def load_mask(mask_path, device):
    totensor = transforms.ToTensor()
    mask = totensor(Image.open(mask_path))
    mask = mask.unsqueeze_(0).float().to(device)
    mask_ori = mask
    mask = F.interpolate(mask, (512,512), mode="nearest")
    mask_small = F.interpolate(mask, (64,64), mode="nearest")
    return mask, mask_small,mask_ori

def save_mask(image_path,mask):
    carvekit = create_carvekit_interface()
    image = Image.open(image_path).convert('RGB').resize((512,512)) 
    image_without_background = carvekit([image])[0]
    image_without_background = np.array(image_without_background)
    est_seg = image_without_background > 127
    image = np.array(image)
    foreground = est_seg[:, : , -1].astype(np.bool_).astype('float')
    mask = mask[0][0].cpu().numpy()
    fg = foreground*mask*255.
    mask_path = image_path.split('.')[0]+'mask.png'
    cv2.imwrite(mask_path, fg.astype('uint8'))

def pil_rectangle_crop(im):
    width, height = im.size   # Get dimensions
    
    if width <= height:
        left = 0
        right = width
        top = (height - width)/2
        bottom = (height + width)/2
    else:
        
        top = 0
        bottom = height
        left = (width - height) / 2
        bottom = (width + height) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im

def add_margin(pil_img, color, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result


def create_carvekit_interface():
    # Check doc strings for more information
    # network unreachable
    interface = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
                            batch_size_seg=5,
                            batch_size_matting=1,
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                            matting_mask_size=2048,
                            trimap_prob_threshold=231,
                            trimap_dilation=30,
                            trimap_erosion_iters=5,
                            fp16=False)


    return interface

def gso_preprocess(interface, input_im):
    '''
    :param input_im (PIL Image).
    :return image (H, W, 3) array in [0, 1].
    '''
    # See https://github.com/Ir1d/image-background-remove-tool
    image = input_im.convert('RGB')
    image = np.array(image)
    image = image[:,90:550,:]
    image = PIL.Image.fromarray(np.array(image))

    # # # forground mask
    # image_without_background = interface([image])[0]
    # image_without_background = np.array(image_without_background)
    # est_seg = image_without_background > 127
    # image = np.array(image)
    # foreground = est_seg[:, : , -1].astype(np.bool_)
    # # image[~foreground] = [255., 255., 255.]

    # # bbox
    # x, y, w, h = cv2.boundingRect(foreground.astype(np.uint8))

    # #image
    # image = image[y:y+h, x:x+w, :]
    # image = PIL.Image.fromarray(np.array(image))
    
    # # resize image such that long edge is 512
    # s = max(image.size)
    # image = add_margin(image, (255//2, 255//2, 255//2), size=s)
    # image = image.resize([400, 400], Image.Resampling.LANCZOS)
    # # image_big = add_margin(image, (255, 255, 255), size=512)
    # image_big = add_margin(image, (255//2, 255//2, 255//2), size=512)
    # image_small = image_big.resize([256, 256], Image.Resampling.LANCZOS)
    # image_small = np.array(image_small)
    image_big = image.resize([512, 512], Image.Resampling.LANCZOS)
    image_small = image_big.resize([256, 256], Image.Resampling.LANCZOS)
    return image_small, image_big


def load_and_preprocess(interface, input_im):
    '''
    :param input_im (PIL Image).
    :return image (H, W, 3) array in [0, 1].
    '''
    # See https://github.com/Ir1d/image-background-remove-tool
    image = input_im.convert('RGB')
    a, b = image.size
    image = add_margin(image, (255, 255, 255), size=a+int(a//2.4))
    # image = add_margin(image, (255, 255, 255), size=a+a//2)
    image_back = copy.deepcopy(np.array(image))
    # forground mask
    image_without_background = interface([image])[0]
    image_without_background = np.array(image_without_background)
    est_seg = image_without_background > 127
    image = np.array(image)
    foreground = est_seg[:, : , -1].astype(np.bool_)
    image[~foreground] = [255., 255., 255.]
    # bbox
    x, y, w, h = cv2.boundingRect(foreground.astype(np.uint8))
    HE, WI = image.shape[0], image.shape[1]
    x2 = WI - x - w
    y2 = HE - y - h 
    s = 2* min([x,y,x2,y2])+min([w,h])
    x_ = x - (s-w)//2
    y_ = y - (s-h)//2

    #image
    image_back = image_back[y_:y_+s, x_:x_+s, :]
    image_back = PIL.Image.fromarray(np.array(image_back))
    image = image[y_:y_+s, x_:x_+s, :]
    image = PIL.Image.fromarray(np.array(image))

    # resize image such that long edge is 512
    image_back = image_back.resize([512, 512], Image.Resampling.LANCZOS)
    image = image.resize([256, 256], Image.Resampling.LANCZOS)

    return image, image_back


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x,torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class AdamWwithEMAandWings(optim.Optimizer):
    # credit to https://gist.github.com/crowsonkb/65f7265353f403714fce3b2595e0b298
    def __init__(self, params, lr=1.e-3, betas=(0.9, 0.999), eps=1.e-8,  # TODO: check hyperparameters before using
                 weight_decay=1.e-2, amsgrad=False, ema_decay=0.9999,   # ema decay to match previous code
                 ema_power=1., param_names=()):
        """AdamW that saves EMA versions of the parameters."""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("Invalid ema_decay value: {}".format(ema_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, ema_decay=ema_decay,
                        ema_power=ema_power, param_names=param_names)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            ema_params_with_grad = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            ema_decay = group['ema_decay']
            ema_power = group['ema_power']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of parameter values
                    state['param_exp_avg'] = p.detach().float().clone()

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                ema_params_with_grad.append(state['param_exp_avg'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            optim._functional.adamw(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=False)

            cur_ema_decay = min(ema_decay, 1 - state['step'] ** -ema_power)
            for param, ema_param in zip(params_with_grad, ema_params_with_grad):
                ema_param.mul_(cur_ema_decay).add_(param.float(), alpha=1 - cur_ema_decay)

        return loss
    
def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image

def preprocess_image(carv, input_im,device):

    small, big = load_and_preprocess(carv, input_im)
    small = transforms.ToTensor()(small).unsqueeze(0).to(device)
    small = small * 2 - 1
    big = transforms.ToTensor()(big).unsqueeze(0).to(device)
    big =big * 2 - 1

    return small, big

@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            c_concat = model.encode_first_stage((input_im.to(c.device))).mode().detach()
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)