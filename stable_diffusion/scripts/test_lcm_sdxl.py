import time  # noqa
import numpy as np  # noqa
from optimum.neuron import NeuronStableDiffusionXLPipeline


# [Export]
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
unet_id = "latent-consistency/lcm-sdxl"
num_images_per_prompt = 1
input_shapes = {"batch_size": 1, "height": 1024, "width": 1024, "num_images_per_prompt": num_images_per_prompt}
compiler_args = {"auto_cast": "matmul", "auto_cast_type": "bf16"}

# Compile and save
stable_diffusion = NeuronStableDiffusionXLPipeline.from_pretrained(
    model_id, unet_id=unet_id, export=True, **compiler_args, **input_shapes
)

save_directory = "lcm_sdxl_neuronx/"
stable_diffusion.save_pretrained(save_directory)

# Push to hub
repo_id = "Jingya/lcm-sdxl-neuronx"
stable_diffusion.push_to_hub(save_directory, repository_id=repo_id, use_auth_token=True)

# # [Inference]
# repo_id = "Jingya/lcm-sdxl-neuronx"
# num_images_per_prompt = 2
# pipe = NeuronStableDiffusionXLPipeline.from_pretrained(repo_id)
# # pipe = NeuronStableDiffusionXLPipeline.from_pretrained(repo_id, data_parallel_mode="unet", num_images_per_prompt=num_images_per_prompt)

# # prompt = ["Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"]*2
# prompt = [
#     "a close-up picture of an old man standing in the rain",
#     "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
# ]

# images = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=8.0).images

# for i in range(5):
#     start_time = time.time()
#     images = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=8.0).images
#     inf_time = time.time() - start_time
#     print(f"[Inference Time] {np.round(inf_time, 2) / num_images_per_prompt} seconds.")

# print(f"Generated {len(images)} images.")

# for i, image in enumerate(images):
#     image.save(f"image_{i}.png")
