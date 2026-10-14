"""DART generator server (InstaFlow 1-step rectified flow + TAESD tiny VAE, 32x32).

Run from this directory (pipeline_rf.py is imported relatively):
    python make_image.py --port 5000

The trainer POSTs a torch tensor (N, 3, 32, 32) in [0, 1] as 'initial_latent' together with a form
field 'transform_strength' (the diffusion timestep T chosen by the reward-guided scheduler), and gets
back a torch tensor of the same shape with the regenerated images.
"""
import argparse
import io

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file
from diffusers import AutoencoderTiny

from pipeline_rf import *

def obs_to_tensors(obs):
    flatten_obs = obs.view(-1, 3, 84, 84)
    flatten_obs = F.interpolate(flatten_obs, size = (32,32), mode='bilinear', align_corners=False)
    return flatten_obs/255.0

app = Flask(__name__)

class PipelineWrapper:
    def __init__(self, device="cuda"):
        self.device = device
        self.pipe =  RectifiedFlowPipeline.from_pretrained("XCLIU/instaflow_0_9B_from_sd_1_5", safety_checker=None,force_download=False,torch_dtype=torch.float16).to("cuda")
        self.pipe.vae = AutoencoderTiny.from_pretrained(
    "sayakpaul/taesd-diffusers", torch_dtype=torch.float16,
).to("cuda")

    def generate_images(self, flatten_obs, transform_strength):
        with torch.no_grad():
            # encode the observations and add noise up to timestep transform_strength
            image=self.pipe.image_processor.preprocess(flatten_obs).to(torch.float16)
            init_latents=self.pipe.vae.config.scaling_factor *retrieve_latents(self.pipe.vae.encode(image),generator=None)
            shape = init_latents.shape
            noise = randn_tensor(shape, generator=None, device="cuda", dtype=torch.float16)
            init_latents = self.pipe.scheduler.add_noise(init_latents, noise, torch.Tensor([transform_strength]).cuda().long())
            # one-step generation from the noised latents
            generated_images = self.pipe(prompt="",
						num_inference_steps=1,
				guidance_scale=0.0,height=32,width=32,
				num_images_per_prompt=init_latents.shape[0],latents=init_latents,
				timesteps=transform_strength,
				output_type='pt').images.to(self.device)

            # generated images in [0, 1] (the trainer rescales to [0, 255])
            background = generated_images.clamp(0, 1).cpu()
            # serialize the tensor
            buffer = io.BytesIO()
            torch.save(background, buffer)
            buffer.seek(0)

            return buffer

pipeline_wrapper = PipelineWrapper()

@app.route('/generate', methods=['POST'])
def generate():
    # load the input tensor from the uploaded file
    initial_latent_file = request.files.get('initial_latent')

    if not initial_latent_file:
        return jsonify({"error": "No initial_latent file provided"}), 400

    initial_latent = torch.load(initial_latent_file).cuda().to(torch.float16)

    # diffusion timestep (defaults to 1 if missing)
    transform_strength = float(request.form.get('transform_strength', 1))

    # generate
    buffer = pipeline_wrapper.generate_images(initial_latent, transform_strength)

    # send the tensor back to the client
    return send_file(buffer, mimetype='application/x-pytorch', as_attachment=True, download_name='generated_images.pt')

parser = argparse.ArgumentParser()
parser.add_argument('--port',default=5000,type=int)
args = parser.parse_args()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port)
