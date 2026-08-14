# app.py - Main Streamlit UI Application (Page Router Edition)
import streamlit as st
import io
import requests
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime

# Import configuration and modules
from config import CUSTOM_CSS, SAMPLE_IMAGES, DEFAULT_MODEL, SUPPORTED_MODELS, ELA_EXPLANATION, FFT_EXPLANATION
from forensics import compute_ela, compute_fft, extract_metadata, generate_forensic_reasons
from models import classify_image

# -------------------------------------------------------------
# App Setup & Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Image Spotter - Real vs. AI Classifier",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom glassmorphism and neon styles
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Helper function to cache sample image downloads
@st.cache_data(show_spinner=True, ttl=3600)
def fetch_sample_image(url: str) -> Image.Image:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        raise RuntimeError(f"Failed to fetch sample image: {e}")

# Helper to resize large images for fast processing
def optimize_image(image: Image.Image, max_dim: int = 1024) -> Image.Image:
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        return image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return image

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------------------------------------
# Sidebar Navigation Menu
# -------------------------------------------------------------
st.sidebar.markdown('<h1>🕵️‍♂️ <span class="neon-text-purple">Forensics Suite</span></h1>', unsafe_allow_html=True)
st.sidebar.write("Advanced Synthetic Media Diagnostics")

# Navigation Menu
nav_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "🔮 AI Image Spotting",
        "📊 Analytics Dashboard",
        "⏳ Forensic History",
        "📁 Batch Processing",
        "📚 Forensic Guide & Math"
    ],
    index=0
)

st.sidebar.markdown("---")

# Model Selection (Available on all pages)
selected_model_id = st.sidebar.selectbox(
    "Deep Learning Model",
    list(SUPPORTED_MODELS.keys()),
    format_func=lambda x: SUPPORTED_MODELS[x]["name"]
)
st.sidebar.caption(SUPPORTED_MODELS[selected_model_id]["description"])

# Quick Gallery Selector (Only visible in Spotting Mode)
if nav_selection == "🔮 AI Image Spotting":
    st.sidebar.markdown("---")
    st.sidebar.write("### 🖼️ Sample Gallery")
    selected_sample = st.sidebar.selectbox(
        "Quick load a test image:",
        ["-- Upload my own --"] + list(SAMPLE_IMAGES.keys())
    )
else:
    selected_sample = "-- Upload my own --"

# -------------------------------------------------------------
# Main Header Section
# -------------------------------------------------------------
st.markdown('<h1>🕵️‍♂️ AI Image <span class="neon-text-blue">Spotter</span></h1>', unsafe_allow_html=True)
st.write("")

# -------------------------------------------------------------
# Page 1: AI Image Spotting
# -------------------------------------------------------------
if nav_selection == "🔮 AI Image Spotting":
    st.markdown("### 🔮 Single Image Verification")
    st.write("Upload an image or load a sample to generate a diagnostic forensic report.")
    
    uploaded_file = st.file_uploader(
        "Drag and drop an image (JPG, JPEG, PNG) to analyze",
        type=["jpg", "jpeg", "png"],
        key="single_uploader"
    )

    active_image = None
    image_source_name = ""

    if selected_sample != "-- Upload my own --":
        try:
            sample_meta = SAMPLE_IMAGES[selected_sample]
            active_image = fetch_sample_image(sample_meta["url"])
            image_source_name = f"Sample: {selected_sample}"
            st.info(f"Loaded {image_source_name} - {sample_meta['description']}")
        except Exception as e:
            st.error(f"Error loading sample: {e}")

    if uploaded_file is not None:
        active_image = Image.open(uploaded_file)
        image_source_name = uploaded_file.name

    if active_image is not None:
        working_image = optimize_image(active_image)
        
        # Run classification and forensics
        with st.spinner("Executing neural model analysis..."):
            classification = classify_image(working_image, selected_model_id)
            
        with st.spinner("Extracting metadata & compression levels..."):
            metadata = extract_metadata(working_image)
            ela_img = compute_ela(working_image, quality=90)
            fft_img = compute_fft(working_image)
            
            # Generate forensic reasons dynamically
            reasons = generate_forensic_reasons(classification, metadata, ela_img, fft_img)

        # Log to history if not already logged (avoid duplication on UI re-run)
        history_filenames = [x["filename"] for x in st.session_state.history]
        # Append only if name is different or history is empty
        if not st.session_state.history or st.session_state.history[-1]["filename"] != image_source_name:
            # Create a memory efficient thumbnail for history view
            history_thumb = working_image.copy()
            history_thumb.thumbnail((300, 300))
            
            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": image_source_name,
                "model": SUPPORTED_MODELS[selected_model_id]["name"],
                "verdict": classification.get("verdict", "ERROR"),
                "fake_score": classification.get("fake_score", 0.5),
                "real_score": classification.get("real_score", 0.5),
                "reasons": reasons,
                "thumbnail": history_thumb
            })

        # Render Verdict Banner
        if classification.get("success", False):
            fake_prob = classification["fake_score"]
            real_prob = classification["real_score"]
            verdict = classification["verdict"]
            
            if verdict == "AI":
                st.markdown(
                    f'<div class="banner-ai"><h3>🚨 VERDICT: AI GENERATED IMAGE DETECTED</h3>'
                    f'<p>The neural network classified this image as synthetic with <b>{fake_prob:.1%}</b> confidence.</p></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="banner-real"><h3>✅ VERDICT: AUTHENTIC PHOTOGRAPH DETECTED</h3>'
                    f'<p>The neural network classified this image as a real photo with <b>{real_prob:.1%}</b> confidence.</p></div>',
                    unsafe_allow_html=True
                )
        else:
            st.error(f"Inference Failure: {classification.get('error')}")
            fake_prob, real_prob, verdict = 0.5, 0.5, "ERROR"

        # Split Dashboard Columns
        col1, col2 = st.columns([5, 5])
        
        with col1:
            st.subheader("🖼️ Analysand Image")
            st.image(active_image, width='stretch', caption=image_source_name)
            
        with col2:
            st.subheader("🧠 Forensic Reasoning Report")
            st.write("Diagnostic indicators analyzed by the forensics pipeline:")
            
            # Render the reasons list as bullet points
            for r in reasons:
                st.write(r)
                
            st.write("")
            st.write("**Neural Confidence Mapping:**")
            m1, m2 = st.columns(2)
            m1.metric("Real Probability", f"{real_prob:.1%}")
            m2.metric("AI Probability", f"{fake_prob:.1%}")
            
            st.progress(fake_prob)

        # Visual Forensic Tabs
        st.write("---")
        st.subheader("🔍 Deep Forensic Toolbox")
        tab_ela, tab_fft, tab_meta, tab_score = st.tabs([
            "🔍 Error Level Analysis (ELA)",
            "📊 FFT Magnitude Spectrum",
            "📁 EXIF Metadata Analysis",
            "📝 Interactive Trust Meter"
        ])
        
        with tab_ela:
            st.markdown('<h4>🔍 Localized Compression Irregularities (ELA)</h4>', unsafe_allow_html=True)
            st.write("Adjust the slider to inspect compression variances. Edges that glow indicate edits or AI patches.")
            ela_quality = st.slider("JPEG Compression Quality Scale", min_value=50, max_value=99, value=90, step=1, key="ela_slider")
            
            with st.spinner("Computing ELA..."):
                ela_visual = compute_ela(working_image, quality=ela_quality)
                
            ela_col1, ela_col2 = st.columns(2)
            ela_col1.image(working_image, width='stretch', caption="Working Copy")
            ela_col2.image(ela_visual, width='stretch', caption=f"ELA Heatmap (Quality={ela_quality})")
            
        with tab_fft:
            st.markdown('<h4>📊 Frequency Domain Artifact Analysis</h4>', unsafe_allow_html=True)
            st.write("Generative networks leave grids in high frequencies. Star-like smooth decays support camera capture.")
            
            fft_col1, fft_col2 = st.columns(2)
            fft_col1.image(working_image, width='stretch', caption="Working Copy")
            fft_col2.image(fft_img, width='stretch', caption="2D FFT Magnitude Spectrum (Inferno Colored)")
            
        with tab_meta:
            st.markdown('<h4>📁 Image Headers & Camera Signature Audit</h4>', unsafe_allow_html=True)
            st.write(f"**Audit Finding:** {metadata['verdict_impact']}")
            
            if metadata["ai_signatures"]:
                st.warning("⚠️ **Warning: AI Software markers found in headers:**")
                for sig in metadata["ai_signatures"]:
                    st.write(f"- {sig}")
                    
            if metadata["has_exif"]:
                with st.expander("Show Complete Raw EXIF Metadata Grid"):
                    st.dataframe(
                        [{"Exif Tag": tag, "Value": val} for tag, val in metadata["details"].items()],
                        width='stretch'
                    )
            else:
                st.write("No EXIF metadata tags found. The image headers are completely empty.")

        with tab_score:
            st.markdown('<h4>📝 Human-in-the-Loop Forensic Trust Meter</h4>', unsafe_allow_html=True)
            st.write("Inspect the image closely and check anomalies to merge human visual judgment with model scores.")
            
            chk_col1, chk_col2 = st.columns(2)
            with chk_col1:
                glitch_hands = st.checkbox("🖐️ Impossible hand/finger shapes")
                glitch_eyes = st.checkbox("👁️ Mismatched irises or pupils")
                glitch_ears = st.checkbox("👂 Asymmetric ears or earrings")
            with chk_col2:
                glitch_text = st.checkbox("🔤 Scrambled background signs/text")
                glitch_bg = st.checkbox("🌀 Melting items or warped geometry")
                glitch_light = st.checkbox("💡 Clashing lighting sources or impossible shadows")
                
            num_checked = sum([glitch_hands, glitch_eyes, glitch_ears, glitch_text, glitch_bg, glitch_light])
            human_penalty = num_checked * 0.15
            combined_ai_score = min(1.0, fake_prob + human_penalty)
            combined_real_score = 1.0 - combined_ai_score
            
            st.markdown("---")
            st.write("### 🚨 Combined Diagnostic Verdict")
            c_m1, c_m2 = st.columns(2)
            c_m1.metric("Combined Real Score", f"{combined_real_score:.1%}")
            c_m2.metric("Combined AI Score", f"{combined_ai_score:.1%}")
            st.progress(combined_ai_score)
            
            if combined_ai_score >= 0.70:
                st.error(f"🔴 **HIGH CRITICAL RISK:** Combined forensics indicate this is highly likely an **AI-generated** or manipulated image (Score: {combined_ai_score:.1%}).")
            elif combined_ai_score >= 0.40:
                st.warning(f"🟡 **SUSPICIOUS:** Moderate indicators suggest this image may contain synthetic patches or generative touch-ups (Score: {combined_ai_score:.1%}).")
            else:
                st.success(f"🟢 **LOW RISK:** Forensic checks align. This image exhibits high parameters of an **authentic photo** (Score: {combined_real_score:.1%}).")
    else:
        st.info("👋 **Welcome to the Spotting Page!** Upload your photo above or load a sample image from the sidebar gallery to begin.")

# -------------------------------------------------------------
# Page 2: Analytics Dashboard
# -------------------------------------------------------------
elif nav_selection == "📊 Analytics Dashboard":
    st.markdown("### 📊 Analytics & Summary Dashboard")
    st.write("Review aggregated stats, distribution metrics, and query logs across all spotting queries.")
    
    if not st.session_state.history:
        st.info("💡 **No logs recorded yet.** Upload and analyze images in the **🔮 AI Image Spotting** page to populate the analytics dashboard.")
    else:
        # Compute Stats
        total_queries = len(st.session_state.history)
        ai_queries = sum(1 for x in st.session_state.history if x["verdict"] == "AI")
        real_queries = sum(1 for x in st.session_state.history if x["verdict"] == "REAL")
        
        # Display Metrics Row
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        d_col1.metric("Total Queries", total_queries)
        d_col2.metric("AI Flagged", ai_queries)
        d_col3.metric("Real Flagged", real_queries)
        d_col4.metric("AI Ratio (%)", f"{ai_queries / total_queries:.1%}" if total_queries > 0 else "0.0%")
        
        st.write("---")
        
        # Render historical log table
        st.subheader("📋 Session Query Log Summary")
        summary_list = []
        for x in st.session_state.history:
            summary_list.append({
                "Timestamp": x["timestamp"],
                "Filename": x["filename"],
                "Model Used": x["model"],
                "AI Verdict": x["verdict"],
                "AI Confidence": f"{x['fake_score']:.1%}",
                "Real Confidence": f"{x['real_score']:.1%}"
            })
        st.dataframe(pd.DataFrame(summary_list), width='stretch')

# -------------------------------------------------------------
# Page 3: Forensic History
# -------------------------------------------------------------
elif nav_selection == "⏳ Forensic History":
    st.markdown("### ⏳ Logged Forensic Query History")
    st.write("Re-inspect and review previous image classifications separately.")
    
    if not st.session_state.history:
        st.info("💡 **History is empty.** Image runs will appear here as soon as they are processed in the Spotting tab.")
    else:
        # Clear buttons
        col_btn1, col_btn2 = st.columns([8, 2])
        with col_btn2:
            if st.button("🧹 Clear Entire History"):
                st.session_state.history = []
                st.rerun()
                
        # Dropdown selection of history entries
        options = [f"[{h['timestamp']}] {h['filename']} (Verdict: {h['verdict']})" for h in st.session_state.history]
        selected_option = st.selectbox("Select a previous log entry to inspect:", options)
        
        selected_index = options.index(selected_option)
        selected_log = st.session_state.history[selected_index]
        
        # Display Details of selected log
        st.write("---")
        st.subheader(f"🔍 Forensic Record: {selected_log['filename']}")
        
        h_col1, h_col2 = st.columns([4, 6])
        
        with h_col1:
            st.image(selected_log["thumbnail"], caption=f"Record Thumbtack - {selected_log['filename']}", use_container_width=True)
            
        with h_col2:
            st.write(f"**Timestamp:** `{selected_log['timestamp']}`")
            st.write(f"**Model Classifier:** `{selected_log['model']}`")
            
            verdict_style = "color:#ef4444;" if selected_log["verdict"] == "AI" else "color:#10b981;"
            st.markdown(f"**Classification Verdict:** <span style='font-size:1.5rem; font-weight:bold; {verdict_style}'>{selected_log['verdict']}</span>", unsafe_allow_html=True)
            st.write(f"**AI Confidence:** `{selected_log['fake_score']:.1%}` | **Real Confidence:** `{selected_log['real_score']:.1%}`")
            
            st.write("---")
            st.write("**🧠 Recorded Reasons List:**")
            for r in selected_log["reasons"]:
                st.write(r)

# -------------------------------------------------------------
# Page 4: Batch Processing Mode
# -------------------------------------------------------------
elif nav_selection == "📁 Batch Processing":
    st.markdown("### 📁 Bulk Forensic Pipeline")
    st.write("Upload folders of files (up to 50+) to evaluate classifications in a single pass.")
    
    uploaded_files = st.file_uploader(
        "Upload multiple images to batch process",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_uploader"
    )
    
    if uploaded_files:
        st.write(f"### ⚙️ Processing {len(uploaded_files)} Images...")
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        results_data = []
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Analyzing {file.name} ({i+1}/{len(uploaded_files)})...")
            try:
                img = Image.open(file)
                # Optimize image size for faster batch prediction
                working_img = optimize_image(img, max_dim=512)
                
                # Predict using the selected model
                classification = classify_image(working_img, selected_model_id)
                metadata = extract_metadata(working_img)
                ela_visual = compute_ela(working_img, quality=90)
                fft_visual = compute_fft(working_img)
                
                # Generate reasons
                reasons = generate_forensic_reasons(classification, metadata, ela_visual, fft_visual)
                
                if classification.get("success", False):
                    fake_p = classification["fake_score"]
                    real_p = classification["real_score"]
                    verdict = classification["verdict"]
                else:
                    fake_p, real_p, verdict = 0.5, 0.5, "ERROR"
                    
                # Store thumbnail
                history_thumb = working_img.copy()
                history_thumb.thumbnail((300, 300))
                
                # Append to history too
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": file.name,
                    "model": SUPPORTED_MODELS[selected_model_id]["name"],
                    "verdict": verdict,
                    "fake_score": fake_p,
                    "real_score": real_p,
                    "reasons": reasons,
                    "thumbnail": history_thumb
                })
                
                results_data.append({
                    "Filename": file.name,
                    "Verdict": verdict,
                    "AI Confidence": f"{fake_p:.1%}",
                    "Real Confidence": f"{real_p:.1%}",
                    "image_obj": img
                })
            except Exception as e:
                results_data.append({
                    "Filename": file.name,
                    "Verdict": "ERROR",
                    "AI Confidence": "N/A",
                    "Real Confidence": "N/A",
                    "image_obj": None
                })
                
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.text(f"✅ Processing complete! Analyzed {len(uploaded_files)} images (added to history).")
        
        # Stats summary
        total_imgs = len(results_data)
        ai_count = sum(1 for x in results_data if x["Verdict"] == "AI")
        real_count = sum(1 for x in results_data if x["Verdict"] == "REAL")
        err_count = sum(1 for x in results_data if x["Verdict"] == "ERROR")
        
        st.write("---")
        st.subheader("📊 Summary Statistics")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Images", total_imgs)
        m_col2.metric("AI Generated", ai_count, delta=f"{ai_count/total_imgs:.0%}" if total_imgs > 0 else "0%")
        m_col3.metric("Real Photos", real_count, delta=f"{real_count/total_imgs:.0%}" if total_imgs > 0 else "0%")
        m_col4.metric("Errors", err_count)
        
        # Display Spreadsheet Table
        df = pd.DataFrame([{
            "Filename": x["Filename"],
            "Verdict": x["Verdict"],
            "AI Confidence": x["AI Confidence"],
            "Real Confidence": x["Real Confidence"]
        } for x in results_data])
        
        st.subheader("📋 Results Spreadsheet")
        st.dataframe(df, width='stretch')
        
        # Export CSV File
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="ai_image_spotter_batch_results.csv",
            mime="text/csv"
        )
        
        # Display Grid View with colored borders
        st.write("---")
        st.subheader("🖼️ Grid Results View")
        st.write("Visual breakdown: Red border represents AI, Green border represents REAL.")
        
        cols_per_row = 4
        for row_idx in range(0, len(results_data), cols_per_row):
            row_items = results_data[row_idx:row_idx+cols_per_row]
            cols = st.columns(cols_per_row)
            for idx, item in enumerate(row_items):
                with cols[idx]:
                    if item["image_obj"] is not None:
                        thumb = item["image_obj"].copy()
                        thumb.thumbnail((200, 200))
                        
                        border_color = "#ef4444" if item["Verdict"] == "AI" else ("#10b981" if item["Verdict"] == "REAL" else "#6b7280")
                        badge_label = "🚨 AI" if item["Verdict"] == "AI" else ("✅ REAL" if item["Verdict"] == "REAL" else "⚠️ ERROR")
                        
                        st.markdown(
                            f'<div style="border: 2px solid {border_color}; border-radius: 8px; padding: 5px; text-align: center; background: rgba(0,0,0,0.15); margin-bottom: 5px;">'
                            f'<span style="font-weight: bold; color: {border_color}; font-size: 0.9rem;">{badge_label}</span><br/>'
                            f'<span style="font-size: 0.75rem; color: #888888; display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{item["Filename"]}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        st.image(thumb, width='stretch')
                    else:
                        st.write("No Image Available")
    else:
        st.info("📁 **Welcome to Batch Mode!** Upload multiple photos to analyze them concurrently and download the CSV prediction logs.")

# -------------------------------------------------------------
# Page 5: Forensic Guide & Math
# -------------------------------------------------------------
elif nav_selection == "📚 Forensic Guide & Math":
    st.markdown("### 📚 Digital Forensics Physics & Mathematical Theory")
    st.write("Detailed explanation of the algorithms powering the classical image checks.")
    
    st.write("---")
    st.markdown(ELA_EXPLANATION)
    st.write("---")
    st.markdown(FFT_EXPLANATION)

# -------------------------------------------------------------
# Footer
# -------------------------------------------------------------
st.write("---")
st.markdown('<div class="footer">AI Image Spotter v1.0.0 | Powered by HuggingFace SigLIP/ViT, NumPy FFT & PIL Forensics</div>', unsafe_allow_html=True)
