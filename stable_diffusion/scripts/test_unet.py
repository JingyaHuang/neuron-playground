import os
from pathlib import Path

from diffusers import DiffusionPipeline, UNet2DConditionModel

from optimum.exporters.neuron import (
    build_stable_diffusion_components_mandatory_shapes,
    export_models,
    get_stable_diffusion_models_for_export,
    infer_stable_diffusion_shapes_from_diffusers,
)
from optimum.exporters.neuron.model_configs import *
from optimum.neuron.utils import (
    DIFFUSION_MODEL_UNET_NAME,
    NEURON_FILE_NAME,
)


# id
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# model_id = "echarlaix/tiny-random-stable-diffusion-xl"
unet_id = "latent-consistency/lcm-sdxl"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
unet = UNet2DConditionModel.from_pretrained(unet_id)
pipe.unet = unet


# prepare neuron config / models
# input_shapes = build_stable_diffusion_components_mandatory_shapes(**{"batch_size": 1, "height": 64, "width": 64})
input_shapes = build_stable_diffusion_components_mandatory_shapes(**{"batch_size": 1, "height": 1024, "width": 1024})
input_shapes = infer_stable_diffusion_shapes_from_diffusers(input_shapes, pipe)
models_and_neuron_configs = get_stable_diffusion_models_for_export(
    pipe,
    task="stable-diffusion-xl",
    dynamic_batch_size=False,
    **input_shapes,
)

# extract only unet
models_and_neuron_configs = {"unet": models_and_neuron_configs["unet"]}
output_model_names = {
    DIFFUSION_MODEL_UNET_NAME: os.path.join(DIFFUSION_MODEL_UNET_NAME, NEURON_FILE_NAME),
}

neuron_inputs, neuron_outputs = export_models(
    models_and_neuron_configs=models_and_neuron_configs,
    output_dir=Path("lcm_sdxl_unet"),
    output_file_names=output_model_names,
    compiler_kwargs={"auto_cast": "matmul", "auto_cast_type": "bf16"},
)

neuron_model = torch.jit.load("lcm_sdxl_unet/unet/model.neuron")
