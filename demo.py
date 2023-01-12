# -- coding: utf-8 --`
import argparse
import os
import json
import random
# engine
from stable_diffusion_engine import StableDiffusionEngine
# scheduler
from diffusers import LMSDiscreteScheduler, PNDMScheduler
# utils
import cv2
import numpy as np
import PIL.Image, PIL.PngImagePlugin
import piexif


EXIF_SOFTWARE_TAG = "stable_diffusion.openvino"

def build_image_metadata(args):
    info = {}
    for name, value in vars(args).items():
        # Special handling for filenames to avoid leaking usernames from paths:
        if name in ['mask', 'init_image', 'output']:
            value = None if value is None else os.path.basename(value)
        if value is not None:
            info[f"stable_diffusion_{name}"] = str(value)

    pnginfo = PIL.PngImagePlugin.PngInfo()
    for key, value in info.items():
        if value is not None:
            pnginfo.add_text(key, value)

    exif_ifd0 = {
        piexif.ImageIFD.Software: EXIF_SOFTWARE_TAG,
        piexif.ImageIFD.ImageDescription: json.dumps(info)
    }
    exif_dict = {
        "0th": exif_ifd0,
    }
    exif_bytes = piexif.dump(exif_dict)

    return dict(pnginfo = pnginfo, exif = exif_bytes)


def read_metadata(source_file):
    try:
        img = PIL.Image.open(source_file)
    except (FileNotFoundError, OSError) as e:
        print(f"Could not open source file to read previous run parameters: {e}")
        raise SystemExit(1)
    if 'exif' not in img.info or not isinstance(img.info['exif'], bytes):
        print(f"No previous run parameters found in file {source_file}")
        raise SystemExit(1)
    try:
        exif = piexif.load(img.info['exif']).get("0th", None)
        assert exif is not None
        assert exif[piexif.ImageIFD.Software] in {EXIF_SOFTWARE_TAG, EXIF_SOFTWARE_TAG.encode('ASCII')}
        return json.loads(exif[piexif.ImageIFD.ImageDescription])
    except Exception as e:
        print(f"Could not decode parameters from previous run in file {source_file}: {e}")
        raise SystemExit(1)
    

def main(args):
    if args.seed is None:
        args.seed = random.randint(0, 2**30)
    np.random.seed(args.seed)
    if args.init_image is None:
        scheduler = LMSDiscreteScheduler(
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            tensor_format="np"
        )
    else:
        scheduler = PNDMScheduler(
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            skip_prk_steps = True,
            tensor_format="np"
        )
    engine = StableDiffusionEngine(
        model = args.model,
        scheduler = scheduler,
        tokenizer = args.tokenizer
    )
    image = engine(
        prompt = args.prompt,
        init_image = None if args.init_image is None else cv2.imread(args.init_image),
        mask = None if args.mask is None else cv2.imread(args.mask, 0),
        strength = args.strength,
        num_inference_steps = args.num_inference_steps,
        guidance_scale = args.guidance_scale,
        eta = args.eta
    )

    pil_image = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


    pil_image.save(args.output,
                   **build_image_metadata(args))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--model", type=str, default="bes-dev/stable-diffusion-v1-4-openvino", help="model name")
    # randomizer params
    parser.add_argument("--seed", type=int, default=None, help="random seed for generating consistent images per prompt")
    # scheduler params
    parser.add_argument("--beta-start", type=float, default=0.00085, help="LMSDiscreteScheduler::beta_start")
    parser.add_argument("--beta-end", type=float, default=0.012, help="LMSDiscreteScheduler::beta_end")
    parser.add_argument("--beta-schedule", type=str, default="scaled_linear", help="LMSDiscreteScheduler::beta_schedule")
    # diffusion params
    parser.add_argument("--num-inference-steps", type=int, default=32, help="num inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument("--eta", type=float, default=0.0, help="eta")
    # tokenizer
    parser.add_argument("--tokenizer", type=str, default="openai/clip-vit-large-patch14", help="tokenizer")
    # prompt
    parser.add_argument("--prompt", type=str, default="Street-art painting of Emilia Clarke in style of Banksy, photorealism", help="prompt")
    # Parameter re-use:
    parser.add_argument("--params-from", type=str, required=False, help="Extract parameters from a previously generated image.")
    # img2img params
    parser.add_argument("--init-image", type=str, default=None, help="path to initial image")
    parser.add_argument("--strength", type=float, default=0.5, help="how strong the initial image should be noised [0.0, 1.0]")
    # inpainting
    parser.add_argument("--mask", type=str, default=None, help="mask of the region to inpaint on the initial image")
    # output name
    parser.add_argument("--output", type=str, default="output.png", help="output image name")
    args = parser.parse_args()

    # Re-calculate args based on 'params_from', if necessary:
    if args.params_from is not None:
        previous_invocation_cmdline = []
        previous_invocation_args = argparse.Namespace()
        original_params = read_metadata(args.params_from)
        for key, value in original_params.items():
            if not key.startswith("stable_diffusion_"):
                continue
            key = key[len("stable_diffusion_"):]
            if key == 'output':
                # Never re-use the output filename from an old invocation.
                continue
            key = f"--{key.replace('_', '-')}"
            previous_invocation_cmdline += [key, value]

        previous_invocation_args = parser.parse_args(previous_invocation_cmdline)
        args = parser.parse_args(namespace = previous_invocation_args)

        print(f"Using arguments from {args.params_from} as base.")
        print("Final arguments are")
        for key in vars(args):
            value = getattr(args, key)
            print(f"  --{key.replace('_', '-')} {value}")

    main(args)