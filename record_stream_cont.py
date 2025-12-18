import cv2
import time
import lancedb
from lancedb.pydantic import LanceModel, Vector
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from sentence_transformers import SentenceTransformer
import os
import torch
import threading
import queue

# --- CONFIGURATION ---
DB_PATH = "data/insight_memory"

# Global queues to pass data between threads
frame_queue = queue.Queue(maxsize=1)  # Holds the latest frame for AI
status_queue = queue.Queue(maxsize=1) # Holds status messages for UI

# --- SCHEMA ---
class InsightLog(LanceModel):
    vector: Vector(384)
    text: str
    timestamp: float
    frame_path: str

# --- THE BRAIN (Runs in background) ---
# --- THE BRAIN (Runs in background) ---
def ai_worker():
    # 1. Update Status: STARTING
    status_queue.put("Init: Checking GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 AI Brain: Using {device.upper()} (this determines speed)")
    
    # 2. Load Vision Model
    status_queue.put("Init: Loading Moondream (Heavy)...")
    print("🧠 AI Brain: Loading Moondream2... (Please wait)")
    
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26"
    try:
        vlm_model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    except Exception as e:
        status_queue.put("Error: Model Load Failed")
        print(f"❌ Critical Error loading AI: {e}")
        return

    # 3. Load Embedder
    status_queue.put("Init: Loading Search Engine...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 4. Load DB
    status_queue.put("Init: Opening Database...")
    os.makedirs(DB_PATH, exist_ok=True)
    db = lancedb.connect(DB_PATH)
    try:
        tbl = db.create_table("logs", schema=InsightLog, exist_ok=True)
    except:
        tbl = db.open_table("logs")

    # 5. READY
    status_queue.put("Ready")
    print("🧠 AI Brain: SYSTEM ONLINE. Ready to record.")
    
    while True:
        try:
            # Wait for a frame (timeout allows checking for exit)
            frame_data = frame_queue.get(timeout=1)
            frame, timestamp = frame_data
            
            # Update status -> THINKING
            status_queue.put("Thinking...")
            
            # Convert to PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Generate Caption
            enc_image = vlm_model.encode_image(pil_image)
            description = vlm_model.answer_question(enc_image, "Describe this image in detail.", tokenizer)
            
            # Save File
            filename = f"captured_frames/frame_{int(timestamp)}.jpg"
            cv2.imwrite(filename, frame)

            # Save to DB
            vector = embed_model.encode(description).tolist()
            tbl.add([{
                "vector": vector, 
                "text": description, 
                "timestamp": timestamp, 
                "frame_path": filename
            }])
            
            # Update status -> DONE (Show first few words)
            print(f"📝 Logged: {description}")
            short_desc = (description[:25] + '..') if len(description) > 25 else description
            status_queue.put(f"Saved: {short_desc}")
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"AI Error: {e}")
            status_queue.put("AI Error (See Terminal)")

# --- THE EYE (Main UI Loop) ---
def main():
    # Start AI Thread
    thread = threading.Thread(target=ai_worker, daemon=True)
    thread.start()

    # Give AI a second to init
    time.sleep(2)

    cap = cv2.VideoCapture(0)
    os.makedirs("captured_frames", exist_ok=True)
    
    last_analysis_time = 0
    analysis_interval = 5  # Seconds
    current_status = "Waiting..."

    print("\n✅ InsightStream Live! (Press 'q' to quit)\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Draw status on screen
        if not status_queue.empty():
            current_status = status_queue.get()
            
        # Add text overlay
        cv2.putText(frame, f"AI Status: {current_status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Send frame to AI every X seconds
        now = time.time()
        if now - last_analysis_time > analysis_interval:
            if frame_queue.empty(): # Only send if AI is free
                frame_queue.put((frame.copy(), now))
                last_analysis_time = now

        cv2.imshow('InsightStream Live', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()