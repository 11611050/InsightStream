<br />
<div align="center">
  <h1 align="center">👁️ InsightStream</h1>

  <p align="center">
    <b>Turn your Webcam into a Searchable Semantic Database with Local AI</b>
  </p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/AI-Moondream2-FF4B4B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Database-LanceDB-000000?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</div>

<br />

---
## 🚀 The Problem vs. The Solution

**The Problem:** Traditional Computer Vision (like YOLO) is "forgetful." It detects a person in a frame, but the moment the person leaves, that information is gone. You cannot ask, *"Who was here 5 minutes ago?"* without complex logging.

**The Solution:** **InsightStream** gives your camera a **Long-Term Memory**. It combines:
1.  **Generative Vision (VLM):** To understand the *context* of a scene, not just bounding boxes.
2.  **Vector Storage (RAG):** To store these understandings as mathematical embeddings.
3.  **Semantic Search:** To let you "talk" to your video feed using natural language.

---

## 🛠️ Tech Stack & Architecture

This project is built entirely on **Open Source** technology and runs **100% Locally** (No API keys required).

| Component | Tech Choice | Why? |
| :--- | :--- | :--- |
| **The Eye** | `OpenCV` | Universal camera support and frame capture. |
| **The Brain** | `Moondream2` | A tiny, powerful Vision-Language Model optimized for edge devices (CPUs). |
| **The Memory** | `LanceDB` | Serverless, multimodal vector database for fast retrieval. |
| **The Logic** | `PyTorch` | Underlying tensor operations for the AI models. |

---

## 💻 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/11611050/InsightStream.git
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    # Activate: 
    # Windows: venv\Scripts\activate
    # Mac/Linux: source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The first run will download ~2GB of AI models automatically)*

---

## 📂 Project Structure

* `record_stream.py`: **(Start Here)** The main snapshot recorder. Recommended for laptops/CPUs.
* `record_stream_cont.py`: Experimental version for powerful GPUs (Continuous recording).
* `ask_stream.py`: The search engine to query your database.
* `requirements.txt`: List of Python libraries needed.

---

## 📸 Usage Guide

### Step 1: Record Memories
Run the recorder to start the camera.
```bash
python record_stream.py
```
* **Action:** Press **`SPACEBAR`** to capture a memory.
* **Status:** Wait for the **"Saved"** message in the terminal/UI before quitting.
* **Quit:** Press **`q`** to exit.

### Step 2: Search Memories
Once you have saved at least one memory, run the query engine:
    ```bash
    python ask_stream.py
    ```
* **Input:** Type a question like *"What am I holding?"* or *"Is there a person?"*
* **Output:** The system will display the most relevant image frame and its timestamp.

## 🔮 Future Roadmap

* **Real-time Voice Interaction:** Integrating `Whisper` so users can ask questions verbally.
* **Web Dashboard:** Building a Streamlit UI to visualize the timeline of memories.
* **Live Alerting:** Send notifications if specific objects (e.g., "Fire", "Stranger") are detected.

---
## 🌟 Let's Connect

I thrive on real business challenges, teamwork, and continuous learning. If you’re seeking a AI Engineer / Data Scientist ready to deliver value from day one, let’s talk!

📮 **Email:** 1997.saiprakash@gmail.com  
🔗**LinkedIN:** [LinkedIn](https://www.linkedin.com/in/sai-prakash-mereddy/)
🔗**HuggingFace:** [Hugging Face](https://huggingface.co/saiprakash97)

---

_Thank you for reviewing my work. Explore the folders above or message me for more details, sample dashboards, or to discuss analytics solutions tailored for your organization._

