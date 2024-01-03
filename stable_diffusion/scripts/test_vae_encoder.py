import os
from pathlib import Path

from diffusers import StableDiffusionPipeline

from optimum.exporters.neuron import (
    build_stable_diffusion_components_mandatory_shapes,
    export_models,
    get_stable_diffusion_models_for_export,
    validate_models_outputs,
)
from optimum.exporters.neuron.__main__ import infer_stable_diffusion_shapes_from_diffusers
from optimum.exporters.neuron.model_configs import *
from optimum.neuron.utils import (
    DIFFUSION_MODEL_VAE_ENCODER_NAME,
    NEURON_FILE_NAME,
)


# id
# model_id = "stabilityai/stable-diffusion-2-1-base"
model_id = "nitrosocke/Ghibli-Diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# prepare neuron config / models
input_shapes = build_stable_diffusion_components_mandatory_shapes(**{"batch_size": 1, "height": 512, "width": 512})
infer_stable_diffusion_shapes_from_diffusers(input_shapes, pipe)
models_and_neuron_configs = get_stable_diffusion_models_for_export(
    pipe,
    task="stable-diffusion",
    dynamic_batch_size=False,
    **input_shapes,
)

# extract only vae encoder
models_and_neuron_configs = {"vae_encoder": models_and_neuron_configs["vae_encoder"]}
output_model_names = {
    DIFFUSION_MODEL_VAE_ENCODER_NAME: os.path.join(DIFFUSION_MODEL_VAE_ENCODER_NAME, NEURON_FILE_NAME),
}

# export
neuron_inputs, neuron_outputs = export_models(
    models_and_neuron_configs=models_and_neuron_configs,
    output_dir=Path("sd_neuron_vae"),
    output_file_names=output_model_names,
    compiler_kwargs={"auto_cast": None},
    # compiler_kwargs={"auto_cast": "matmul", "auto_cast_type": "bf16"},
)

validate_models_outputs(
    models_and_neuron_configs=models_and_neuron_configs,
    neuron_named_outputs=neuron_outputs,
    output_dir=Path("sd_neuron_vae"),
    neuron_files_subpaths=output_model_names,
)
