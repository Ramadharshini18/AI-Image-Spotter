# AI Image Spotting Portal 🖼️🛡️

A premium, modern web application designed to analyze image pixel grids, metadata structures, and compression error levels to classify and spot AI-generated images vs. real camera photographs.

Built with a **Decoupled FastAPI Python Backend** (powered by the **SigLIP Transformer model**) and a **Crisp Light React + Vite Frontend**.

---

## 🎨 UI Features
* **Hybrid Clean Layout**: An attractive, professional light mode workspace with high-contrast slate navy typography.
* **3-Column Result Dashboard**:
  * **Left**: Active Image thumbnail preview.
  * **Middle**: Instant Verdict banner (AI vs. REAL) and probability meters.
  * **Right**: Reasoning Report displaying neural audit details.
* **Scan History Registry**: Persistent local session log to review, inspect, or delete previous queries.
* **Batch processing**: Bulk upload module to audit multiple files simultaneously.

---

## 🚀 How to Run Locally

### 1. Backend Server Setup
1. Navigate to the project root:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the FastAPI development server:
   ```bash
   python server.py
   ```
   *The server runs at: http://127.0.0.1:8000*

### 2. Frontend Portal Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies and start the Vite server:
   ```bash
   npm install
   npm run dev
   ```
   *The client portal runs at: http://localhost:5173*

---

## 📦 Production Deployment

To deploy this application to the cloud (Render, Heroku, or Docker) without any **CORS** or **API Port connection errors**, follow these steps:

### Step A: Compile the Frontend React Assets
From the root folder, compile your static React assets:
```bash
npm run build --prefix frontend
```
*This compiles the portal and saves the files into `frontend/dist/`.*

### Step B: Run the Combined Server
Once built, FastAPI is configured to host the static assets directly. Just run your python app:
```bash
python server.py
```
FastAPI will now serve both the React website on the root path `/` and all API endpoints on the same port, ensuring same-origin mapping and eliminating CORS configurations!
