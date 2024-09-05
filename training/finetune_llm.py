from optimum.neuron import NeuronTrainer as Trainer
from optimum.neuron.distributed import lazy_load_for_parallelism

# Define the tensor_parallel_size
tensor_parallel_size = 8

# Load model from the Hugging face Hub 
with lazy_load_for_parallelism(tensor_parallel_size=tensor_parallel_size):
    model = AutoModelForCausalLM.from_pretrained(model_id)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    data_collator=default_data_collator,  # no special collator needed since we stacked the dataset
)

# Start training
trainer.train()

trainer.save_model()  # saves the tokenizer too for easy upload