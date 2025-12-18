from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import time

print("--- DIAGNOSTIC MODE ---")

# 1. Force CPU to be safe
device = "cpu"
print(f"1. Using Device: {device}")

# 2. Load Model
print("2. Loading Model... (This might take a minute)")
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        revision=revision
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    print("✅ Model Loaded Successfully.")
except Exception as e:
    print(f"❌ Model Load Failed: {e}")
    exit()

# 3. Create a Dummy Image (Red Square)
print("3. Creating Test Image...")
img = Image.new('RGB', (500, 500), color='red')

# 4. Run Inference
print("4. Testing 'Thinking' Speed...")
start_time = time.time()

try:
    print("   ... AI is thinking (Please wait) ...")
    enc_image = model.encode_image(img)
    answer = model.answer_question(enc_image, "Describe this image.", tokenizer)
    
    end_time = time.time()
    print(f"\n✅ SUCCESS! The AI spoke.")
    print(f"📝 Output: {answer}")
    print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")

except Exception as e:
    print(f"\n❌ AI CRASHED during thinking: {e}")