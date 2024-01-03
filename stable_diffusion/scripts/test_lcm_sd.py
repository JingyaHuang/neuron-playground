from optimum.neuron import NeuronLatentConsistencyModelPipeline


# [Export]
# model_id = "SimianLuo/LCM_Dreamshaper_v7"
model_id = "echarlaix/tiny-random-latent-consistency"
num_images_per_prompt = 4
input_shapes = {"batch_size": 1, "height": 64, "width": 64, "num_images_per_prompt": num_images_per_prompt}
# input_shapes = {"batch_size": 1, "height": 768, "width": 768, "num_images_per_prompt": num_images_per_prompt}
compiler_args = {"auto_cast": "matmul", "auto_cast_type": "bf16"}

# Compile and save
stable_diffusion = NeuronLatentConsistencyModelPipeline.from_pretrained(
    model_id, export=True, compiler_workdir="sd_neuron_lcm_neff/", **compiler_args, **input_shapes
)

# save_directory = "lcm_768_neuronx/"
# stable_diffusion.save_pretrained(save_directory)

# # Push to hub
# repo_id = "Jingya/LCM_Dreamshaper_v7_neuronx"
# stable_diffusion.push_to_hub(save_directory, repository_id=repo_id, use_auth_token=True)

# # [Inference]
# repo_id = "Jingya/LCM_Dreamshaper_v7_neuronx"
# pipe = NeuronLatentConsistencyModelPipeline.from_pretrained(repo_id)

# # prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
# prompt = [
#     "a close-up picture of an old man standing in the rain",
#     "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
# ]

# images = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=8.0).images

# for i in range(5):
#     start_time = time.time()
#     images = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=8.0).images
#     inf_time = time.time() - start_time
#     print(f"[Inference Time] {np.round(inf_time, 2)} seconds.")
#     print(f"Generated {len(images)} images.")


# images[0].save("image.png")
