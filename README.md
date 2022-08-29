# stable_diffusion.openvino

Implementation of Text-To-Image generation using Stable Diffusion on Intel CPU.
<p align="center">
  <img src="data/title.png"/>
</p>

## Requirements

* Linux, Windows, MacOS
* Python 3.8.+
* CPU compatible with OpenVINO.

## Install requirements

```bash
pip install -r requirements.txt
```

## Generate image from text description

```bash
usage: stable_diffusion.py [-h] [--models-cfg MODELS_CFG] [--beta-start BETA_START] [--beta-end BETA_END] [--beta-schedule BETA_SCHEDULE] [--num-inference-steps NUM_INFERENCE_STEPS]
                           [--guidance-scale GUIDANCE_SCALE] [--eta ETA] [--tokenizer TOKENIZER] [--prompt PROMPT] [--output OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  --models-cfg MODELS_CFG
                        path to models config
  --beta-start BETA_START
                        LMSDiscreteScheduler::beta_start
  --beta-end BETA_END   LMSDiscreteScheduler::beta_end
  --beta-schedule BETA_SCHEDULE
                        LMSDiscreteScheduler::beta_schedule
  --num-inference-steps NUM_INFERENCE_STEPS
                        num inference steps
  --guidance-scale GUIDANCE_SCALE
                        guidance scale
  --eta ETA             eta
  --tokenizer TOKENIZER
                        tokenizer
  --prompt PROMPT       prompt
  --output OUTPUT       output image name
```

### Example
```bash
python stable_diffusion.py --prompt "Street-art painting of Emilia Clarke in style of Banksy, photorealism"
```

## Acknowledgements

* Original implementation of Stable Diffusion: https://github.com/CompVis/stable-diffusion
* diffusers library: https://github.com/huggingface/diffusers

## Disclaimer

The authors are not responsible for the content generated using this project.
Please, don't use this project to produce illegal, harmful, offensive etc. content.
