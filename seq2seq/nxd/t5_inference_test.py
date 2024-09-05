import torch
from transformers import T5ForConditionalGeneration
import t5_models  
import neuronx_distributed
import time 


model_name = "google/flan-t5-xl" 
max_length = 128
num_beams = 4
tp_degree = 8 # tensor parallelism degree

# model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto")
# torch.save({"model":model.state_dict()}, model_name.split("/")[-1] + ".pt")
# model.config.use_cache = True


if __name__ == '__main__':
    # This can take up to 20 minutes
    encoder_compile_start_time = time.time()
    traced_encoder = t5_models.parallel_trace_encoder(model_name, max_length, num_beams, tp_degree)
    print("Encoder compilation time {}".format(time.time() - encoder_compile_start_time))
    neuronx_distributed.trace.parallel_model_save(traced_encoder, "TracedParallelEncoder.pt")
    
    # This can take up to 15 minutes
    # decoder_compile_start_time = time.time()
    # traced_decoder = t5_models.parallel_trace_decoder(model, model_name, num_beams, max_length, tp_degree)
    # print("Decoder compilation time {}".format(time.time() - decoder_compile_start_time))    
    # neuronx_distributed.trace.parallel_model_save(traced_decoder, "TracedParallelDecoder.pt")