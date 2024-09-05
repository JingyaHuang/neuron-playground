import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import neuronx_distributed
import time 
import t5_models  
from wrapper import T5Wrapper


model_name = "google/flan-t5-xl" 
max_length = 128
num_beams = 4
tp_degree = 8 # tensor parallelism degree

model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto")
# torch.save({"model":model.state_dict()}, model_name.split("/")[-1] + ".pt")
# model.config.use_cache = True


if __name__ == '__main__':
    # [Compile Encoder] This can take up to 20 minutes
    # encoder_compile_start_time = time.time()
    # traced_encoder = t5_models.parallel_trace_encoder(model_name, max_length, num_beams, tp_degree)
    # print("Encoder compilation time {}".format(time.time() - encoder_compile_start_time))
    # neuronx_distributed.trace.parallel_model_save(traced_encoder, "TracedParallelEncoder.pt")
    
    # # [Compile Decoder] This can take up to 15 minutes
    # decoder_compile_start_time = time.time()
    # traced_decoder = t5_models.parallel_trace_decoder(model, model_name, num_beams, max_length, tp_degree)
    # print("Decoder compilation time {}".format(time.time() - decoder_compile_start_time))    
    # neuronx_distributed.trace.parallel_model_save(traced_decoder, "TracedParallelDecoder.pt")
    
    # [Inference]
    num_return_sequences = 4

    traced_encoder = neuronx_distributed.trace.parallel_model_load("TracedParallelEncoder.pt")
    traced_decoder = neuronx_distributed.trace.parallel_model_load("TracedParallelDecoder.pt")

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5Wrapper.from_pretrained(model_name)

    model.encoder = traced_encoder
    model.decoder = traced_decoder
    setattr(model.encoder, 'main_input_name', 'input_ids')  # Attribute required by beam search

    output = model.parallel_infer(tokenizer=tokenizer,
                                prompt="translate English to German: Lets eat good food.",
                                max_length=max_length,
                                num_beams=num_beams,
                                num_return_sequences=num_return_sequences,
                                device="xla")

    results = [tokenizer.decode(t, skip_special_tokens=True) for t in output]

    print('Results:')
    for i, summary in enumerate(results):
        print(i + 1, summary)
    