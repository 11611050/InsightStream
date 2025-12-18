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

# Queues
frame_queue = queue.Queue(maxsize=1)
status_queue = queue.Queue(maxsize=1)

# Schema
class InsightLog(LanceModel):
    vector: Vector(384)
    text: str
    timestamp: float
    frame_path: str

# --- THE BRAIN (Background Worker) ---
def ai_worker():
    status_queue.put("Init: Loading AI Models...")
    print("🧠 AI Brain: Loading... (Please wait)")
    
    device = "cpu" # Forced CPU based on your diagnostic
    
    # Load VLM
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26"
    try:
        vlm_model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    except Exception as e:
        status_queue.put("Error: AI Load Failed")
        return

    # Load Search
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load DB
    os.makedirs(DB_PATH, exist_ok=True)
    db = lancedb.connect(DB_PATH)
    try:
        tbl = db.create_table("logs", schema=InsightLog, exist_ok=True)
    except:
        tbl = db.open_table("logs")

    status_queue.put("Ready: Press SPACE to Capture")
    print("🧠 AI Brain: Ready.")
    
    while True:
        try:
            # Wait for frame
            frame_data = frame_queue.get(timeout=1)
            frame, timestamp = frame_data
            
            # Update Status
            print("\n👀 AI is thinking... (This will take ~2 mins)")
            status_queue.put("Thinking... (Wait ~120s)")
            
            # Process
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Generate Caption
            enc_image = vlm_model.encode_image(pil_image)
            description = vlm_model.answer_question(enc_image, "Describe this image in detail.", tokenizer)
            
            # Save
            filename = f"captured_frames/frame_{int(timestamp)}.jpg"
            cv2.imwrite(filename, frame)

            vector = embed_model.encode(description).tolist()
            tbl.add([{
                "vector": vector, 
                "text": description, 
                "timestamp": timestamp, 
                "frame_path": filename
            }])
            
            # Done
            print(f"✅ SAVED: {description}")
            status_queue.put("Saved! Ready for next.")
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error: {e}")
            status_queue.put("Error processing image")

# --- THE EYE (Main UI) ---
def main():
    thread = threading.Thread(target=ai_worker, daemon=True)
    thread.start()

    cap = cv2.VideoCapture(0)
    os.makedirs("captured_frames", exist_ok=True)
    current_status = "Initializing..."

    print("\n📷 Snapshot Mode Active")
    print("--------------------------------")
    print("PRESS 'SPACEBAR' to capture a memory.")
    print("PRESS 'q' to quit.")
    print("--------------------------------\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Update Status Text
        if not status_queue.empty():
            current_status = status_queue.get()

        # Display UI
        # Black bar at top for text
        cv2.rectangle(frame, (0,0), (640, 50), (0,0,0), -1)
        cv2.putText(frame, f"Status: {current_status}", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('InsightStream Camera', frame)

        key = cv2.waitKey(1) & 0xFF
        
        # SPACEBAR to Capture
        if key == 32: 
            if "Thinking" not in current_status and "Init" not in current_status:
                current_status = "Capturing..."
                frame_queue.put((frame.copy(), time.time()))
            else:
                print("⚠️ System is busy! Wait for it to finish.")

        # Q to Quit
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()