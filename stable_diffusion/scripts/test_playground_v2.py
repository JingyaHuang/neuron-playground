import time  # noqa
import numpy as np  # noqa
from optimum.neuron import NeuronStableDiffusionXLPipeline
from diffusers import EulerAncestralDiscreteScheduler


# # [Export]
# model_id = "stabilityai/sdxl-turbo"
# num_images_per_prompt = 1
# input_shapes = {"batch_size": 1, "height": 512, "width": 512, "num_images_per_prompt": num_images_per_prompt}
# compiler_args = {"auto_cast": "matmul", "auto_cast_type": "bf16"}

# # Compile and save
# stable_diffusion = NeuronStableDiffusionXLPipeline.from_pretrained(
#     model_id, export=True, **compiler_args, **input_shapes
# )

# save_directory = "playground_v2_neuron/"
# stable_diffusion = NeuronStableDiffusionXLPipeline.from_pretrained(save_directory)
# # stable_diffusion.save_pretrained(save_directory)

# # Push to hub
# repo_id = "Jingya/playground-v2-neuronx"
# stable_diffusion.push_to_hub(save_directory, repository_id=repo_id, use_auth_token=True)

# [Inference]
repo_id = "Jingya/playground-v2-neuronx"
num_images_per_prompt = 1
pipe = NeuronStableDiffusionXLPipeline.from_pretrained(
    repo_id, data_parallel_mode="all", num_images_per_prompt=num_images_per_prompt
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
# pipe = NeuronStableDiffusionXLPipeline.from_pretrained(repo_id, data_parallel_mode="unet", num_images_per_prompt=num_images_per_prompt)

prompt = ["Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"] * 2
# prompt = [
#     "a close-up picture of an old man standing in the rain",
#     "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
# ]

images = pipe(prompt=prompt).images

for i in range(5):
    start_time = time.time()
    images = pipe(prompt=prompt, guidance_scale=7.5).images
    inf_time = time.time() - start_time
    print(f"[Inference Time] {np.round(inf_time, 2) / num_images_per_prompt} seconds.")
    images[0].save(f"image_{i}.png")

print(f"Generated {len(images)} images.")
