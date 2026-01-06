import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

# ------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------
# Set this to None to load the Base Model (Comparison Step)
# Set this to "trained" AFTER you train (Final Step)
MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct" 
# MODEL_PATH = "trained" #

max_seq_length = 2048
dtype = None
load_in_4bit = True

# ------------------------------------------------------------------------
# 1. LOAD MODEL
# ------------------------------------------------------------------------
print(f"⏳ Loading model: {MODEL_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# ------------------------------------------------------------------------
# 2. CHAT LOOP
# ------------------------------------------------------------------------
print("\n🤖 Model Loaded! Type 'exit' to quit.")
print("------------------------------------------------------------------")

# Define the System Prompt (The Persona)
system_prompt = """You are the helpful host of 'Arpita Farmstay' in Malvan, India. 
Answer questions accurately based on the house rules and local tips.
If you don't know the answer, say so politely."""

messages = [] # Keep history if you want, or reset per loop

while True:
    user_input = input("\n👤 You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Prepare the specific prompt structure for Qwen
    # Note: We reset messages every turn to keep it simple for testing knowledge
    # (prevents the model from hallucinating based on previous turns)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    print("🤖 Assistant: ", end="")
    
    # TextStreamer prints token-by-token (Matrix style)
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    _ = model.generate(
        input_ids = inputs, 
        streamer = streamer, 
        max_new_tokens = 256, 
        use_cache = True,
        temperature = 0.7
    )
    print("") # Newline