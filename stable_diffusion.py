import argparse
import os
import inspect
import numpy as np
# openvino
from openvino.runtime import Core
# tokenizer
from transformers import CLIPTokenizer
# scheduler
from diffusers import LMSDiscreteScheduler
# utils
from tqdm import tqdm
import cv2
import gdown
import json


def gdrive_file(name, url, md5):
    print(f"load file: {name}...")
    ckpt_path = os.path.join("data", name)
    gdown.cached_download(url, ckpt_path, md5=md5)
    return ckpt_path


class StableDiffusion:
    def __init__(
            self,
            models_cfg,
            scheduler,
            tokenizer="openai/clip-vit-large-patch14",
            device="CPU"
    ):
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
        self.scheduler = scheduler
        # models
        self.core = Core()
        # text features
        self._text_encoder = self.core.read_model(
            gdrive_file("text_encoder.xml", models_cfg["text_encoder.xml"]["url"], models_cfg["text_encoder.xml"]["md5"]),
            gdrive_file("text_encoder.bin", models_cfg["text_encoder.bin"]["url"], models_cfg["text_encoder.bin"]["md5"])
        )
        self.text_encoder = self.core.compile_model(self._text_encoder, device)
        # diffusion
        self._unet = self.core.read_model(
            gdrive_file("unet.xml", models_cfg["unet.xml"]["url"], models_cfg["unet.xml"]["md5"]),
            gdrive_file("unet.bin", models_cfg["unet.bin"]["url"], models_cfg["unet.bin"]["md5"])
        )
        self.unet = self.core.compile_model(self._unet, device)
        self.latent_shape = tuple(self._unet.inputs[0].shape)[1:]
        # decoder
        self._vae = self.core.read_model(
            gdrive_file("vae.xml", models_cfg["vae.xml"]["url"], models_cfg["vae.xml"]["md5"]),
            gdrive_file("vae.bin", models_cfg["vae.bin"]["url"], models_cfg["vae.bin"]["md5"])
        )
        self.vae = self.core.compile_model(self._vae, device)

    def __call__(self, prompt, num_inference_steps = 32, guidance_scale = 7.5, eta = 0.0):
        result = lambda var: next(iter(var.values()))

        # extract condition
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True
        ).input_ids
        text_embeddings = result(
            self.text_encoder.infer_new_request({"tokens": np.array([tokens])})
        )

        # do classifier free guidance
        if guidance_scale > 1.0:
            tokens_uncond = self.tokenizer(
                "",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True
            ).input_ids
            uncond_embeddings = result(
                self.text_encoder.infer_new_request({"tokens": np.array([tokens_uncond])})
            )
            text_embeddings = np.concatenate((uncond_embeddings, text_embeddings), axis=0)

        # make noise
        latents = np.random.randn(*self.latent_shape)

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.stack([latents, latents], 0) if guidance_scale > 1.0 else latents
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = result(self.unet.infer_new_request({
                "latent_model_input": latent_model_input,
                "t": t,
                "encoder_hidden_states": text_embeddings
            }))

            # perform guidance
            if guidance_scale > 1.0:
                noise_pred = noise_pred[0] + guidance_scale * (noise_pred[1] - noise_pred[0])

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs)["prev_sample"]
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

        image = result(self.vae.infer_new_request({
            "latents": np.expand_dims(latents, 0)
        }))

        # convert tensor to opencv's image format
        image = (image / 2 + 0.5).clip(0, 1)
        image = (image[0].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
        return image


def main(args):
    scheduler = LMSDiscreteScheduler(
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        tensor_format="np"
    )
    stable_diffusion = StableDiffusion(
        models_cfg = json.load(open(args.models_cfg)),
        scheduler = scheduler,
        tokenizer = args.tokenizer
    )
    image = stable_diffusion(
        prompt = args.prompt,
        num_inference_steps = args.num_inference_steps,
        guidance_scale = args.guidance_scale,
        eta = args.eta
    )
    cv2.imwrite(args.output, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--models-cfg", type=str, default="data/models.json", help="path to models config")
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
    # output name
    parser.add_argument("--output", type=str, default="output.png", help="output image name")
    args = parser.parse_args()
    main(args)
