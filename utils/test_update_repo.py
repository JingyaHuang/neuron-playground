from optimum.neuron import NeuronStableDiffusionXLPipeline


# [BERT]
# MODEL_ID = "hf-internal-testing/tiny-random-BertModel"
# NEURON_MODEL_REPO = "optimum/tiny_random_bert_neuronx"
# STATIC_INPUTS_SHAPES = {"batch_size": 1, "sequence_length": 32}

# neuron_model = NeuronModelForSequenceClassification.from_pretrained(MODEL_ID, export=True, **STATIC_INPUTS_SHAPES)
# save_directory = "bert_neuron_tiny/"
# neuron_model.save_pretrained(save_directory)
# neuron_model.push_to_hub(save_directory, repository_id=NEURON_MODEL_REPO, use_auth_token=True)

# [SDXL]
MODEL_ID = "echarlaix/tiny-random-stable-diffusion-xl"
NEURON_MODEL_REPO = "Jingya/tiny-random-stable-diffusion-xl-neuronx"
COMPILER_ARGS = {"auto_cast": "all", "auto_cast_type": "bf16"}
STATIC_INPUTS_SHAPES = {
    "height": 64,  # width of the image
    "width": 64,  # height of the image
    "num_images_per_prompt": 1,  # number of images to generate per prompt
    "batch_size": 1,  # batch size for the model
}

neuron_model = NeuronStableDiffusionXLPipeline.from_pretrained(
    MODEL_ID, export=True, **STATIC_INPUTS_SHAPES, **COMPILER_ARGS
)
save_directory = "sdxl_neuron_tiny"
neuron_model.save_pretrained(save_directory)
neuron_model.push_to_hub(save_directory, repository_id=NEURON_MODEL_REPO, use_auth_token=True)
