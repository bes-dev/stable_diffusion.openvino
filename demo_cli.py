# -- coding: utf-8 --`
import argparse
import os
# engine
from stable_diffusion_engine import StableDiffusionEngine
# scheduler
from diffusers import LMSDiscreteScheduler, PNDMScheduler
# utils
import cv2
import numpy as np

import time
from datetime import datetime


def outputCurrentSettings(settings):
    print("model: ", settings.model)
    print("tokenizer: ", settings.tokenizer)
    print("(s)seed: ", settings.seed)
    print("(n)number of inferece steps: ", settings.num_inference_steps)
    print("(g)guidance scale or weight of prompt: ", settings.guidance_scale)
    print("(p)prompt: ", settings.prompt)
    print("(i)init image path: ", settings.init_image)
    print("(is)init image strength: ", settings.strength)
    print("(ma)mask for inpainting file path : ", settings.mask)
    print("(o)output path: ", settings.output)
    print("")


def askForAction():
    print("Please enter one of the ()s above to set new value, e.g.: p")
    print("And enter an empty line to run the generation with what's set!")
    print("input: ", end='')
    return input()

def askForArgument(action):
    print()

def buildNewOutputPathWithTime(path):
    timeSt = time.time()
    lastDotInd = path.rfind('.')
    if(lastDotInd < 0):
        return path + "_" + str(timeSt)

    newPath = path[0:lastDotInd] + "_" + str(timeSt) + "_" +path[lastDotInd:]

    # if the original path does not have an .extention, then add it:
    if (os.path.basename(newPath).find(str(timeSt)) < 0):
        newPath = path + "_" + str(timeSt)
    return newPath


def performGeneration(engine, settings):
    print("Started generation at ", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    image = engine(
        prompt = settings.prompt,
        init_image = None if settings.init_image is None else cv2.imread(settings.init_image),
        mask = None if settings.mask is None else cv2.imread(settings.mask, 0),
        strength = settings.strength,
        num_inference_steps = settings.num_inference_steps,
        guidance_scale = settings.guidance_scale,
        eta = settings.eta
    )
    cv2.imwrite(buildNewOutputPathWithTime(settings.output), image)
    print("Finished generation at ", datetime.now().strftime("%Y/%m/%d %H:%M:%S"), end='\n\n\n')


def initializeEngine(settings):

    if settings.seed is not None:
        np.random.seed(settings.seed)

    if settings.init_image is None:
        scheduler = LMSDiscreteScheduler(
            beta_start=settings.beta_start,
            beta_end=settings.beta_end,
            beta_schedule=settings.beta_schedule,
            tensor_format="np"
        )
    else:
        scheduler = PNDMScheduler(
            beta_start=settings.beta_start,
            beta_end=settings.beta_end,
            beta_schedule=settings.beta_schedule,
            skip_prk_steps = True,
            tensor_format="np"
        )

    print('Started initializing the Engine at ', datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    engine = StableDiffusionEngine(
        model = settings.model,
        scheduler = scheduler,
        tokenizer = settings.tokenizer
    )
    print('Done initializing the Engine at ', datetime.now().strftime("%Y/%m/%d %H:%M:%S"), end='\n\n')

    return engine

def main(settings):
    engine = initializeEngine(settings)

    availableActions = ['s', 'n', 'g', 'p', 'i', 'is', 'ma', 'o']

    while(True):
        outputCurrentSettings(settings)
        action = askForAction()

        if (action == ""):
            performGeneration(engine, settings)
            continue

        try:
            # throws if no such setting found
            availableActions.index(action)
        except:
            print('\n\nNo such setting. Please try again.\n\n')
            continue


        print('Input new value: ', end='')
        newValue = input()

        if(action == 's'):
            settings.seed = int(newValue)
        elif(action == 'n'):
            settings.num_inference_steps = int(newValue)
        elif(action == 'g'):
            settings.guidance_scale = float(newValue)
        elif(action == 'p'):
            settings.prompt = newValue
        elif(action == 'i'):
            settings.init_image = newValue
        elif(action == 'is'):
            settings.strength = float(newValue)
        elif(action == 'ma'):
            settings.mask = newValue
        elif(action == 'o'):
            settings.output = newValue

        print('New value has been set!', end="\n\n\n")


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
    # img2img params
    parser.add_argument("--init-image", type=str, default=None, help="path to initial image")
    parser.add_argument("--strength", type=float, default=0.5, help="how strong the initial image should be noised [0.0, 1.0]")
    # inpainting
    parser.add_argument("--mask", type=str, default=None, help="mask of the region to inpaint on the initial image")
    # output name
    parser.add_argument("--output", type=str, default="out/output.png", help="output image name")
    args = parser.parse_args()
    main(args)
