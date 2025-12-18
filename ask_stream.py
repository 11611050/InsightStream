import lancedb
from sentence_transformers import SentenceTransformer
from lancedb.pydantic import LanceModel, Vector
import cv2
import os
import datetime

# --- CONFIGURATION ---
DB_PATH = "data/insight_memory"

# --- SCHEMA (Must match the recorder) ---
class InsightLog(LanceModel):
    vector: Vector(384)
    text: str
    timestamp: float
    frame_path: str

# 1. Load Resources
print("--- Loading Search Engine ---")
db = lancedb.connect(DB_PATH)
try:
    tbl = db.open_table("logs")
except:
    print("Error: No data found! Run record_stream.py first.")
    exit()

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def search_memory(query):
    print(f"Searching for: '{query}'...")
    
    # Convert text query to vector
    query_vector = embed_model.encode(query).tolist()
    
    # Search LanceDB (Semantic Search)
    # limit(3) means "Show me the top 3 matches"
    results = tbl.search(query_vector).limit(3).to_list()
    
    if not results:
        print("No matches found.")
        return

    print(f"\n✅ Found {len(results)} matches:\n")
    
    for i, res in enumerate(results):
        # Convert timestamp to readable time
        time_str = datetime.datetime.fromtimestamp(res['timestamp']).strftime('%H:%M:%S')
        print(f"Match {i+1} [{time_str}]: {res['text']}")
        
        # Show the image evidence
        if os.path.exists(res['frame_path']):
            img = cv2.imread(res['frame_path'])
            
            # Add text to image
            cv2.putText(img, f"{time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(f"Match {i+1}", img)
    
    print("\n(Press any key on the image windows to close them...)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("--- InsightStream Query Engine Ready ---")
    while True:
        user_q = input("\nAsk InsightStream (or 'exit'): ")
        if user_q.lower() == 'exit': break
        
        search_memory(user_q)