import os
import tempfile

from optimum.neuron import NeuronStableDiffusionPipeline


model_id = "hf-internal-testing/tiny-stable-diffusion-torch"
compiler_args = {"auto_cast": "matmul", "auto_cast_type": "bf16"}
input_shapes = {"batch_size": 1, "height": 64, "width": 64}

with tempfile.TemporaryDirectory() as tempdir:
    save_path = f"{tempdir}/neff"
    stable_diffusion = NeuronStableDiffusionPipeline.from_pretrained(
        model_id, export=True, optlevel="3", compiler_workdir=save_path, **compiler_args, **input_shapes
    )
    import pdb

    pdb.set_trace()
    if not os.path.isdir(save_path):
        raise

# save_directory = "sd_neuron_tiny/"
# stable_diffusion.save_pretrained(save_directory)
