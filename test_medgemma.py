import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_PATH = "./models/medgemma-1.5-4b-it"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Loading MedGemma...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="mps",
)

model.eval()

print("Model device:", next(model.parameters()).device)
print("MPS available:", torch.backends.mps.is_available())

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "What are the major risk factors for coronary artery disease? "
                    "Give a concise clinical explanation."
                ),
            }
        ],
    }
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(
    text=prompt,
    return_tensors="pt",
)

inputs = {
    k: v.to("mps") if hasattr(v, "to") else v
    for k, v in inputs.items()
}

print("Generating...")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
    )

input_length = inputs["input_ids"].shape[-1]

generated_tokens = output[:, input_length:]

response = processor.batch_decode(
    generated_tokens,
    skip_special_tokens=True,
)[0]

print("\n===== MEDGEMMA RESPONSE =====\n")
print(response)