"""
natural-image morphing in the latent space of a diffusion model (Stable Diffusion 1.5)

    1. encode images A and B with the VAE,
    2. DDIM-invert both latents (empty prompt, no guidance) up to a fraction `depth`
       of the diffusion trajectory,
    3. slerp between the two inverted latents with weight w,
    4. DDIM-denoise back to an image (deterministic: same w -> same image).

`depth` sets the trade-off: low -> faithful to A/B but closer to a cross-fade,
high -> natural-looking but invented intermediate content.

Requires the optional dependencies:  pip install "physion[diffusion]"
(~2 GB model download on first use). Generated images are cached on disk.
"""
import os
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler

from .images import normalize
from .spectrum import equalize_pair
from .morphing import pair_weights

MODEL_ID = 'stable-diffusion-v1-5/stable-diffusion-v1-5'


def default_device():
    return 'mps' if torch.backends.mps.is_available() else \
        'cuda' if torch.cuda.is_available() else 'cpu'


def slerp(a, b, w):
    af, bf = a.flatten(), b.flatten()
    omega = torch.arccos(torch.clamp(torch.dot(af / af.norm(), bf / bf.norm()), -1, 1))
    if omega < 1e-4:
        return (1 - w) * a + w * b
    return (torch.sin((1 - w) * omega) * a + torch.sin(w * omega) * b) / torch.sin(omega)


class DiffusionMorpher:
    """
    imgs:       array of images (n, height, width) in [0,1]
    gen_shape:  (h, w) generation resolution, multiples of 64
    spectrum:   amplitude-spectrum equalization within a pair, see spectrum.equalize_pair
    cache_dir:  generated images are stored there (None: no cache)
    """

    def __init__(self, imgs, model_id=MODEL_ID, gen_shape=(512, 640),
                 n_steps=25, depth=0.7, prompt='', spectrum='radial',
                 cache_dir=None, device=None, **kwargs):
        self.imgs = imgs
        self.shape = imgs[0].shape
        self.gen_shape, self.n_steps, self.depth = gen_shape, n_steps, depth
        self.spectrum = spectrum
        self.kwargs = kwargs            # of `normalize`: mean_lum, rms_contrast
        self.device = device or default_device()
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, variant='fp16', torch_dtype=torch.float32,
            safety_checker=None, requires_safety_checker=False).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.sched = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.sched.set_timesteps(n_steps)
        self.n = int(round(depth * n_steps))
        self.step = self.sched.config.num_train_timesteps // n_steps
        self.alphas = self.sched.alphas_cumprod.to(self.device)
        self.emb = self.pipe.encode_prompt(prompt, self.device, 1, False)[0]
        self.scale = self.pipe.vae.config.scaling_factor
        self._inverted = {}
        self.cache_dir = cache_dir
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    @torch.no_grad()
    def encode(self, img):
        im = Image.fromarray(np.round(255 * img).astype(np.uint8))
        im = im.resize(self.gen_shape[::-1], Image.BICUBIC)
        x = torch.tensor(np.asarray(im, dtype=np.float32) / 127.5 - 1.)
        x = x[None, None].repeat(1, 3, 1, 1).to(self.device)
        return self.pipe.vae.encode(x).latent_dist.mean * self.scale

    @torch.no_grad()
    def invert(self, i):
        """ deterministic DDIM inversion of image i up to `depth` """
        if i not in self._inverted:
            z = self.encode(self.imgs[i])
            for t in self.sched.timesteps.flip(0)[:self.n]:
                t_prev = t - self.step
                a_prev = self.alphas[t_prev] if t_prev >= 0 else \
                    self.sched.final_alpha_cumprod.to(self.device)
                eps = self.pipe.unet(z, t, encoder_hidden_states=self.emb).sample
                x0 = (z - (1 - a_prev).sqrt() * eps) / a_prev.sqrt()
                z = self.alphas[t].sqrt() * x0 + (1 - self.alphas[t]).sqrt() * eps
            self._inverted[i] = z
        return self._inverted[i]

    @torch.no_grad()
    def denoise(self, z):
        for t in self.sched.timesteps[self.n_steps - self.n:]:
            eps = self.pipe.unet(z, t, encoder_hidden_states=self.emb).sample
            z = self.sched.step(eps, t, z).prev_sample
        x = self.pipe.vae.decode(z / self.scale).sample
        return ((x.mean(1)[0].clamp(-1, 1) + 1) / 2).cpu().numpy()

    def generate(self, i, j, w):
        """ raw generated image (original resolution, [0,1]) between image i (w=0) and j (w=1) """
        if i > j:
            i, j, w = j, i, 1. - w
        fn = None if self.cache_dir is None else os.path.join(
            self.cache_dir, '%i-%i_w-%.4f_depth-%.2f_steps-%i.npy' % (i, j, w, self.depth, self.n_steps))
        if fn is not None and os.path.isfile(fn):
            x = np.load(fn).astype(np.float32)
        else:
            x = self.denoise(slerp(self.invert(i), self.invert(j), w))
            if fn is not None:
                np.save(fn, x.astype(np.float16))
        x = Image.fromarray(x.astype(np.float32))
        return np.asarray(x.resize(self.shape[::-1], Image.BICUBIC), dtype=float)

    def morph(self, i, j, w):
        return normalize(self.generate(i, j, w), **self.kwargs)

    def pair(self, i, j, s_raw, anchored=False):
        """ two stimuli with raw similarity s_raw, see morphing.pair_weights """
        w1, w2 = pair_weights(s_raw, anchored)
        return equalize_pair(self.generate(i, j, w1), self.generate(i, j, w2),
                             mode=self.spectrum, **self.kwargs)
