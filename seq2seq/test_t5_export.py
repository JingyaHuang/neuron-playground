from transformers import AutoTokenizer

from optimum.exporters.neuron.model_configs import *
from optimum.neuron import NeuronModelForSeq2SeqLM


# id
# model_id = "t5-small"
# model = T5ForConditionalGeneration.from_pretrained(model_id)

# [Encoder]
# neuron_config = T5EncoderNeuronConfig(
#     config=model.config, task="text2text-generation", dynamic_batch_size=True, batch_size=1, sequence_length=18, num_beams=4
# )

# neuron_inputs, neuron_outputs = export(
#     model=model,
#     config=neuron_config,
#     output=Path("t5_small_encoder/model.neuron"),
#     auto_cast = "matmul",
#     auto_cast_type = "bf16",
# )

# [Decoder]
# neuron_config = T5DecoderNeuronConfig(
#     config=model.config,
#     task="text2text-generation",
#     dynamic_batch_size=True,
#     batch_size=1,
#     sequence_length=18,
#     num_beams=4,
# )
# neuron_inputs, neuron_outputs = export(
#     model=model,
#     config=neuron_config,
#     output=Path("t5_small_decoder/model.neuron"),
#     auto_cast="matmul",
#     auto_cast_type="bf16",
# )

# # [Export All]
# input_shapes = {
#     "batch_size": 2,
#     "sequence_length": 64,
#     # "num_beams": 4,
#     "num_beams": 1,
# }
# # model_id = "hf-internal-testing/tiny-random-t5"
# model_id = "t5-small"
# neuron_model = NeuronModelForSeq2SeqLM.from_pretrained(model_id, export=True, dynamic_batch_size=False, output_attentions=True, output_hidden_states=True, **input_shapes)
# # save_path = "t5_small_tiny/"
# save_path = "t5_small_neuron/"
# # save_path = "t5_small_neuron_beam_4"
# neuron_model.save_pretrained(save_path)
# # repository_id = "Jingya/tiny-random-t5-neuronx"
# # neuron_model.push_to_hub(save_path, repository_id=repository_id, use_auth_token=True)
# del neuron_model

save_path = "t5_small_neuron"
# save_path = "t5_small_neuron_beam_4"
neuron_model = NeuronModelForSeq2SeqLM.from_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(save_path)
prompt = ["translate English to German: Lets eat good food.", "translate English to German: good night my friends."]
inputs = tokenizer(prompt, padding=True, return_tensors="pt")
# num_return_sequences = 2

output = neuron_model.generate(
    **inputs,
    # num_return_sequences=num_return_sequences,
    output_attentions=True,
    output_hidden_states=True,
    output_scores=True,
    # max_length=3,
    # max_new_tokens=5,
    return_dict_in_generate=False,
)

results = [tokenizer.decode(t, skip_special_tokens=True) for t in output]

print("Results:")
for i, summary in enumerate(results):
    print(i + 1, summary)
