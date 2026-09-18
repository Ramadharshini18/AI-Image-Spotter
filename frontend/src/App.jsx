// frontend/src/App.jsx
import React, { useState, useEffect } from 'react';
import { 
  Shield, 
  ShieldAlert, 
  History, 
  Layers, 
  BookOpen, 
  UploadCloud, 
  RefreshCw, 
  CheckSquare, 
  Trash2, 
  Download, 
  ChevronDown,
  ChevronUp,
  Info,
  Image as ImageIcon 
} from 'lucide-react';

const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? "http://127.0.0.1:8000"
  : "";
const MODEL_ID = "Ateeqq/ai-vs-human-image-detector";

// Constants from config.py
const SAMPLE_IMAGES = {
  "Real Human Portrait": {
    "url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?q=80&w=500",
    "type": "REAL",
    "description": "A high-resolution photograph of a person with natural skin textures, hair strands, and lighting reflections."
  },
  "Real Nature Landscape": {
    "url": "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?q=80&w=500",
    "type": "REAL",
    "description": "Real nature photography showing complex natural fractals and light dispersion."
  },
  "AI Generated Landscape": {
    "url": "https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/images/stable_diffusion_15.png",
    "type": "AI",
    "description": "Stable Diffusion generated artistic landscape. Look for smooth blending and geometric inconsistencies."
  },
  "AI Generated Portrait": {
    "url": "https://raw.githubusercontent.com/taki0112/Diffusion-Models-in-Practice/main/assets/sd_sample.png",
    "type": "AI",
    "description": "AI-generated face. Common artifacts include blended ear shapes and mismatched background patterns."
  }
};

const SUPPORTED_MODELS = {
  "Ateeqq/ai-vs-human-image-detector": {
    "name": "SigLIP High-Res Classifier (Recommended)",
    "description": "SigLIP model fine-tuned on 120,000 high-resolution images (60k Real, 60k AI). Highly accurate on photographs."
  }
};

const compressImageBase64 = (base64Str, maxWidth = 120, quality = 0.6) => {
  return new Promise((resolve) => {
    if (!base64Str || base64Str.startsWith("http")) {
      resolve(base64Str);
      return;
    }
    const img = new Image();
    img.src = base64Str;
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      
      let width = img.width;
      let height = img.height;
      
      if (width > maxWidth) {
        height = Math.round((height * maxWidth) / width);
        width = maxWidth;
      }
      
      canvas.width = width;
      canvas.height = height;
      ctx.drawImage(img, 0, 0, width, height);
      
      const compressed = canvas.toDataURL("image/jpeg", quality);
      resolve(compressed);
    };
    img.onerror = () => {
      resolve(base64Str);
    };
  });
};

export default function App() {
  const [page, setPage] = useState("spotting");
  
  // History list backed by LocalStorage
  const [history, setHistory] = useState(() => {
    try {
      const saved = localStorage.getItem("forensic_history");
      if (saved && saved.length > 2.5 * 1024 * 1024) {
        console.warn("Cached history is too large. Clearing to prevent quota issues.");
        localStorage.removeItem("forensic_history");
        return [];
      }
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      console.warn("Could not read forensic history from LocalStorage:", e);
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem("forensic_history", JSON.stringify(history));
    } catch (e) {
      console.warn("LocalStorage quota exceeded. Evicting older entries to save space.", e);
      if (history.length > 5) {
        setHistory(prev => prev.slice(0, prev.length - 5));
      }
    }
  }, [history]);

  // --- Single Spotting State ---
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [activeImageUri, setActiveImageUri] = useState("");
  const [activeImageFile, setActiveImageFile] = useState(null);
  const [activeImageUrl, setActiveImageUrl] = useState("");
  const [imageName, imageNameSet] = useState("");
  const [activeTab, setActiveTab] = useState("ela");
  const [elaQuality, setElaQuality] = useState(90);
  const [recalculatingEla, setRecalculatingEla] = useState(false);
  const [customElaUri, setCustomElaUri] = useState("");
  const [showTechDetails, setShowTechDetails] = useState(false);
  const [showHistoryTechDetails, setShowHistoryTechDetails] = useState(false);
  
  // Trust Checklist anomalies state
  const [anomalies, setAnomalies] = useState({
    hands: false,
    eyes: false,
    ears: false,
    text: false,
    bg: false,
    light: false
  });

  // Reset checklist when result changes
  useEffect(() => {
    setAnomalies({
      hands: false,
      eyes: false,
      ears: false,
      text: false,
      bg: false,
      light: false
    });
    setElaQuality(90);
    setCustomElaUri("");
    setShowTechDetails(false);
  }, [result]);

  // Recalculate combined scores based on checkboxes
  const fakeProb = result ? result.fake_score : 0.5;
  const numChecked = Object.values(anomalies).filter(Boolean).length;
  const humanPenalty = numChecked * 0.15;
  const combinedAiScore = Math.min(1.0, fakeProb + humanPenalty);
  const combinedRealScore = 1.0 - combinedAiScore;

  // Single Image API Upload
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Set preview
    const reader = new FileReader();
    reader.onload = () => {
      setActiveImageUri(reader.result);
    };
    reader.readAsDataURL(file);
    
    imageNameSet(file.name);
    setActiveImageFile(file);
    setActiveImageUrl("");
    setResult(null);
  };

  // URL Gallery Fetch Predict
  const handleLoadSample = async (sampleKey, sampleData) => {
    imageNameSet(`Sample: ${sampleKey}`);
    setActiveImageUri(sampleData.url);
    setActiveImageFile(null);
    setActiveImageUrl(sampleData.url);
    setResult(null);
  };

  // Trigger analysis on submit button click
  const triggerAnalysis = async () => {
    if (!activeImageUri) return;
    
    setAnalyzing(true);
    setResult(null);
    
    try {
      if (activeImageFile) {
        const formData = new FormData();
        formData.append("file", activeImageFile);
        
        const response = await fetch(`${API_BASE}/api/analyze?model_id=${MODEL_ID}`, {
          method: "POST",
          body: formData
        });
        const data = await response.json();
        if (response.ok) {
          setResult(data);
          await logQueryToHistory(data, activeImageUri);
        } else {
          alert(`Analysis failed: ${data.detail || "Unknown error"}`);
        }
      } else if (activeImageUrl) {
        const response = await fetch(`${API_BASE}/api/analyze-url?model_id=${MODEL_ID}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: activeImageUrl })
        });
        const data = await response.json();
        if (response.ok) {
          setResult(data);
          await logQueryToHistory(data, activeImageUrl);
        } else {
          alert(`Analysis failed: ${data.detail || "Unknown error"}`);
        }
      }
    } catch (err) {
      alert(`API Connection error: ${err.message}`);
    } finally {
      setAnalyzing(false);
    }
  };

  // ELA Slider API Recalculation
  const handleElaSlider = async (val) => {
    setElaQuality(val);
    setRecalculatingEla(true);
    try {
      const response = await fetch(`${API_BASE}/api/ela?quality=${val}`, {
        method: "POST"
      });
      const data = await response.json();
      if (response.ok) {
        setCustomElaUri(data.ela_base64);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setRecalculatingEla(false);
    }
  };

  const logQueryToHistory = async (data, thumbUri) => {
    let compressedThumb = thumbUri;
    try {
      compressedThumb = await compressImageBase64(thumbUri, 120, 0.65);
    } catch (e) {
      console.warn("Thumbnail compression failed, caching original.", e);
    }
    const newLog = {
      id: Date.now(),
      timestamp: new Date().toLocaleTimeString() + " " + new Date().toLocaleDateString(),
      filename: data.filename,
      model: "SigLIP High-Res Classifier",
      verdict: data.verdict,
      verdict_label: data.verdict_label || (data.verdict === 'AI' ? 'Likely AI-Generated' : 'Likely Real'),
      summary: data.summary || (data.diagnostic_report?.verdict_banner?.summary ?? ""),
      fake_score: data.fake_score,
      real_score: data.real_score,
      reasons: data.reasons,
      diagnostic_report: data.diagnostic_report,
      thumbnail: compressedThumb
    };
    setHistory(prev => [newLog, ...prev]);
  };

  // --- Batch State ---
  const [batching, setBatching] = useState(false);
  const [batchResults, setBatchResults] = useState([]);
  
  // Batch API Upload
  const handleBatchUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    
    setBatching(true);
    setBatchResults([]);
    
    const results = [];
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      // Create reader preview
      const previewPromise = new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.readAsDataURL(file);
      });
      const thumbUri = await previewPromise;
      
      const formData = new FormData();
      formData.append("file", file);
      
      try {
        const response = await fetch(`${API_BASE}/api/analyze?model_id=${MODEL_ID}`, {
          method: "POST",
          body: formData
        });
        const data = await response.json();
        if (response.ok) {
          results.push({
            id: Date.now() + i,
            filename: file.name,
            verdict: data.verdict,
            verdict_label: data.verdict_label || (data.verdict === 'AI' ? 'Likely AI-Generated' : 'Likely Real'),
            fake_score: data.fake_score,
            real_score: data.real_score,
            thumbnail: thumbUri
          });
          // Also log batch runs into general query history
          await logQueryToHistory(data, thumbUri);
        } else {
          results.push({
            id: Date.now() + i,
            filename: file.name,
            verdict: "ERROR",
            fake_score: 0.5,
            real_score: 0.5,
            thumbnail: thumbUri
          });
        }
      } catch (err) {
        results.push({
          id: Date.now() + i,
          filename: file.name,
          verdict: "CONNECTION ERROR",
          fake_score: 0.5,
          real_score: 0.5,
          thumbnail: thumbUri
        });
      }
      setBatchResults([...results]); // update visual progress
    }
    setBatching(false);
  };

  // Export Batch Results to CSV
  const handleExportCSV = () => {
    let csvContent = "data:text/csv;charset=utf-8,Filename,Verdict,AI Confidence,Real Confidence\n";
    batchResults.forEach(item => {
      const vLabel = item.verdict_label || (item.verdict === 'AI' ? 'Likely AI-Generated' : (item.verdict === 'REAL' ? 'Likely Real' : item.verdict));
      csvContent += `"${item.filename}","${vLabel}","${(item.fake_score*100).toFixed(1)}%","${(item.real_score*100).toFixed(1)}%"\n`;
    });
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "ai_image_spotter_batch_results.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // --- Inspect Log State ---
  const [inspectLog, setInspectLog] = useState(null);
  
  // Set default inspector to first item if present
  useEffect(() => {
    if (history.length && !inspectLog) {
      setInspectLog(history[0]);
    }
  }, [history, inspectLog]);

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <div className="sidebar">
        <div className="logo-section">
          <Shield size={28} />
          <h2>Spotter Suite</h2>
        </div>
        
        <div className="nav-menu">
          <button 
            className={`nav-item ${page === 'spotting' ? 'active' : ''}`}
            onClick={() => setPage("spotting")}
          >
            <Shield size={18} />
            AI Image Spotting
          </button>

          
          <button 
            className={`nav-item ${page === 'history' ? 'active' : ''}`}
            onClick={() => setPage("history")}
          >
            <History size={18} />
            Scan History
          </button>
          
          <button 
            className={`nav-item ${page === 'batch' ? 'active' : ''}`}
            onClick={() => setPage("batch")}
          >
            <Layers size={18} />
            Batch Processing
          </button>

        </div>
      </div>

      {/* Main Content Area */}
      <div className="main-content">

        {/* -------------------------------------------------------------
            PAGE 1: SPOTTING
           ------------------------------------------------------------- */}
        {page === 'spotting' && (
          <div>
            <div className="page-header">
              <h1>🔮 AI Image Spotting</h1>
              <p>Upload a photo or choose a sample image to audit its synthetic pixel signatures.</p>
            </div>

            <div className={result ? "grid-cols-3" : "grid-cols-2"}>
              {/* Column 1: Input Box / Active Image */}
              <div>
                {!activeImageUri && (
                  <div className="card">
                    <h3 style={{ marginBottom: '1rem' }}>🖼️ Upload Target</h3>
                    <label className="dropzone">
                      <UploadCloud size={48} style={{ color: 'var(--primary-purple)' }} />
                      <div>
                        <p>Drag and drop image here or click to browse</p>
                        <span style={{ fontSize: '0.8rem', color: 'var(--text-slate)' }}>Supports PNG, JPG, JPEG</span>
                      </div>
                      <input type="file" onChange={handleFileUpload} style={{ display: 'none' }} accept="image/*" />
                    </label>
                    
                    <h4>🖼️ Or Load Gallery Sample:</h4>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.75rem' }}>
                      {Object.entries(SAMPLE_IMAGES).map(([name, data]) => (
                        <button 
                          key={name}
                          className="btn-select"
                          onClick={() => handleLoadSample(name, data)}
                        >
                          {name}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {activeImageUri && (
                  <div className="card" style={{ textAlign: 'center' }}>
                    <h3 style={{ marginBottom: '1rem', textAlign: 'left' }}>📷 Active Image</h3>
                    <div className="image-preview-box" style={{ marginBottom: '1rem' }}>
                      <img src={activeImageUri} alt="Active preview" />
                    </div>
                    <span style={{ display: 'block', marginBottom: '1.25rem', fontSize: '0.85rem', color: 'var(--text-slate)' }}>
                      Filename: {imageName}
                    </span>
                    
                    {result ? (
                      <button 
                        className="btn-purple" 
                        style={{ width: '100%', padding: '0.9rem', fontSize: '1.05rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
                        onClick={() => {
                          setActiveImageUri("");
                          setActiveImageFile(null);
                          setActiveImageUrl("");
                          setResult(null);
                        }}
                      >
                        <UploadCloud size={18} />
                        <span>Upload Another Image</span>
                      </button>
                    ) : (
                      <button 
                        className="btn-purple" 
                        style={{ width: '100%', padding: '0.9rem', fontSize: '1.05rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
                        onClick={triggerAnalysis}
                        disabled={analyzing}
                      >
                        {analyzing ? (
                          <>
                            <div className="loader" style={{ width: '16px', height: '16px', borderTopColor: '#ffffff' }}></div>
                            <span>Detecting Image...</span>
                          </>
                        ) : (
                          <>
                            <Shield size={18} />
                            <span>Detect Image</span>
                          </>
                        )}
                      </button>
                    )}
                  </div>
                )}
              </div>

              {/* Column 2: Detection & Probability when predicted, otherwise loaders/empty state */}
              {!result ? (
                <div>
                  {analyzing && (
                    <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '1rem' }}>
                      <div className="loader" style={{ width: '40px', height: '40px' }}></div>
                      <p style={{ fontWeight: 600, color: 'var(--primary-purple)', textAlign: 'center' }}>Analyzing pixel matrices via SigLIP Transformer...</p>
                    </div>
                  )}

                  {!analyzing && (
                    <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '1rem', color: 'var(--text-slate)', textAlign: 'center' }}>
                      <ImageIcon size={48} style={{ opacity: 0.3 }} />
                      <p>No image analyzed yet. Drop an image or select a sample to begin.</p>
                    </div>
                  )}
                </div>
              ) : (
                <div>
                  {result.verdict === 'AI' ? (
                    <div className="verdict-banner ai">
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <ShieldAlert size={24} />
                        <h3>VERDICT: LIKELY AI-GENERATED</h3>
                      </div>
                      <p>{result.summary || `Our AI model found visual patterns that are more consistent with AI-generated images (${(result.fake_score*100).toFixed(1)}% confidence).`}</p>
                    </div>
                  ) : (
                    <div className="verdict-banner real">
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Shield size={24} />
                        <h3>VERDICT: LIKELY REAL</h3>
                      </div>
                      <p>{result.summary || `Our AI model found visual patterns that are more consistent with a natural photograph than with synthetic images (${(result.real_score*100).toFixed(1)}% confidence).`}</p>
                    </div>
                  )}

                  <div className="card" style={{ padding: '1.5rem' }}>
                    <h3 style={{ marginBottom: '1rem' }}>📊 Prediction Probability</h3>
                    
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                      <div className="meter-section">
                        <div className="meter-header">
                          <span>Real Probability</span>
                          <span style={{ color: 'var(--accent-real)' }}>{(result.real_score*100).toFixed(1)}%</span>
                        </div>
                        <div className="meter-track">
                          <div className="meter-fill real" style={{ width: `${result.real_score*100}%` }}></div>
                        </div>
                      </div>

                      <div className="meter-section">
                        <div className="meter-header">
                          <span>AI Probability</span>
                          <span style={{ color: 'var(--primary-purple)' }}>{(result.fake_score*100).toFixed(1)}%</span>
                        </div>
                        <div className="meter-track">
                          <div className="meter-fill ai" style={{ width: `${result.fake_score*100}%` }}></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Column 3: Diagnostic Report when predicted */}
              {result && (
                <div>
                  <div className="card" style={{ margin: 0, height: '100%' }}>
                    <h3>🧠 Why did we reach this verdict?</h3>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-slate)', marginBottom: '1.25rem' }}>
                      Explainable evidence evaluated for {result.filename}:
                    </p>

                    {result.diagnostic_report?.contradiction_note && (
                      <div className="contradiction-notice">
                        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                          <Info size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
                          <span>{result.diagnostic_report.contradiction_note}</span>
                        </div>
                      </div>
                    )}

                    <div className="reasons-list">
                      {result.diagnostic_report?.evidence_cards ? (
                        result.diagnostic_report.evidence_cards.map((card, i) => (
                          <div key={i} className={`reason-item-card ${card.status_type || 'neutral'}`}>
                            <span className="reason-icon">{card.icon}</span>
                            <div className="reason-content">
                              <div className="evidence-header">
                                <strong className="reason-label" style={{ marginBottom: 0 }}>{card.title}</strong>
                                <span className={`status-badge ${card.status_type || 'neutral'}`}>
                                  {card.status}
                                </span>
                              </div>
                              <p className="reason-detail">{card.explanation}</p>
                            </div>
                          </div>
                        ))
                      ) : (
                        result.reasons?.map((r, i) => {
                          const isObj = typeof r === 'object' && r !== null;
                          const type = isObj ? (r.type || 'neutral') : 'neutral';
                          const icon = isObj ? r.icon : 'ℹ️';
                          const label = isObj ? r.label : 'Diagnostic Audit';
                          const detail = isObj ? r.detail : String(r).replace(/\*\*/g, '');
                          
                          return (
                            <div key={i} className={`reason-item-card ${type}`}>
                              <span className="reason-icon">{icon}</span>
                              <div className="reason-content">
                                <strong className="reason-label">{label}</strong>
                                <p className="reason-detail">{detail}</p>
                              </div>
                            </div>
                          );
                        })
                      )}
                    </div>

                    {result.diagnostic_report?.technical_details && (
                      <div>
                        <button
                          type="button"
                          className="tech-toggle-btn"
                          onClick={() => setShowTechDetails(prev => !prev)}
                        >
                          <span>View Technical Details</span>
                          {showTechDetails ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                        </button>

                        {showTechDetails && (
                          <div className="tech-drawer">
                            <div className="tech-grid">
                              <div className="tech-metric-card">
                                <span className="tech-metric-title">Model</span>
                                <span className="tech-metric-value" style={{ fontSize: '0.92rem' }}>
                                  {result.diagnostic_report.technical_details.model_name || 'SigLIP Classifier'}
                                </span>
                                <span className="tech-metric-sub">
                                  AI: {(result.fake_score * 100).toFixed(1)}% | Real: {(result.real_score * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="tech-metric-card">
                                <span className="tech-metric-title">Error Level Analysis</span>
                                <span className="tech-metric-value">
                                  {result.diagnostic_report.technical_details.ela_std}
                                </span>
                                <span className="tech-metric-sub">
                                  Threshold: &gt; {result.diagnostic_report.technical_details.ela_threshold} std
                                </span>
                              </div>
                              <div className="tech-metric-card">
                                <span className="tech-metric-title">FFT Frequency</span>
                                <span className="tech-metric-value">
                                  {result.diagnostic_report.technical_details.fft_std}
                                </span>
                                <span className="tech-metric-sub">
                                  Threshold: &gt; {result.diagnostic_report.technical_details.fft_threshold} std
                                </span>
                              </div>
                              <div className="tech-metric-card">
                                <span className="tech-metric-title">Metadata Audit</span>
                                <span className="tech-metric-value">
                                  {result.diagnostic_report.technical_details.has_exif ? `${result.diagnostic_report.technical_details.metadata_count} tags` : 'No EXIF'}
                                </span>
                                <span className="tech-metric-sub">
                                  {result.diagnostic_report.technical_details.ai_signatures?.length > 0 
                                    ? `AI Sigs: ${result.diagnostic_report.technical_details.ai_signatures.join(', ')}`
                                    : 'No direct AI tags'}
                                </span>
                              </div>
                            </div>

                            {result.diagnostic_report.technical_details.metadata_tags && Object.keys(result.diagnostic_report.technical_details.metadata_tags).length > 0 && (
                              <div style={{ marginTop: '0.75rem', maxHeight: '180px', overflowY: 'auto' }}>
                                <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-slate)' }}>Raw Metadata Headers:</span>
                                <table className="exif-table" style={{ marginTop: '0.35rem', fontSize: '0.8rem' }}>
                                  <thead>
                                    <tr>
                                      <th style={{ padding: '0.4rem 0.6rem' }}>Tag</th>
                                      <th style={{ padding: '0.4rem 0.6rem' }}>Value</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {Object.entries(result.diagnostic_report.technical_details.metadata_tags).map(([k, v]) => (
                                      <tr key={k}>
                                        <td style={{ padding: '0.4rem 0.6rem', fontWeight: 600 }}>{k}</td>
                                        <td style={{ padding: '0.4rem 0.6rem' }}>{String(v)}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}


        {/* -------------------------------------------------------------
            PAGE 3: HISTORY
           ------------------------------------------------------------- */}
        {page === 'history' && (
          <div>
            <div className="page-header">
              <h1>⏳ Scan History</h1>
              <p>Select and review logged images separately to inspect their results.</p>
            </div>

            {history.length === 0 ? (
              <div className="card" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-slate)' }}>
                <ImageIcon size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                <p>Scan history is empty. Logs will appear here as soon as you evaluate files.</p>
              </div>
            ) : (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem', alignItems: 'center' }}>
                  <h3>📜 Scan Registry</h3>
                  <button 
                    className="btn-purple" 
                    style={{ backgroundColor: 'var(--accent-ai)', boxShadow: 'none' }}
                    onClick={() => {
                      if (confirm("Delete all query history?")) {
                        setHistory([]);
                        setInspectLog(null);
                      }
                    }}
                  >
                    <Trash2 size={16} style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
                    Clear Registry Logs
                  </button>
                </div>

                <div className="grid-cols-2">
                  {/* Left History Column: List */}
                  <div className="card" style={{ maxHeight: '600px', overflowY: 'auto' }}>
                    <h4 style={{ marginBottom: '1rem' }}>Logged Queries:</h4>
                    {history.map((log) => (
                      <div 
                        key={log.id} 
                        className={`nav-item ${inspectLog?.id === log.id ? 'active' : ''}`}
                        style={{ display: 'flex', gap: '1rem', padding: '0.75rem', marginBottom: '0.5rem', cursor: 'pointer', borderRadius: '8px' }}
                        onClick={() => setInspectLog(log)}
                      >
                        <img 
                          src={log.thumbnail} 
                          alt="Thumb" 
                          style={{ width: '40px', height: '40px', objectFit: 'cover', borderRadius: '4px' }} 
                        />
                        <div style={{ flexGrow: 1, minWidth: 0 }}>
                          <p style={{ fontSize: '0.85rem', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {log.filename}
                          </p>
                          <span style={{ fontSize: '0.75rem', opacity: 0.8 }}>{log.timestamp}</span>
                        </div>
                        <span style={{ 
                          fontSize: '0.8rem', 
                          fontWeight: 700, 
                          color: log.verdict === 'AI' ? (inspectLog?.id === log.id ? '#ffffff' : 'var(--accent-ai)') : (inspectLog?.id === log.id ? '#ffffff' : 'var(--accent-real)')
                        }}>
                          {log.verdict_label || (log.verdict === 'AI' ? 'Likely AI-Generated' : 'Likely Real')}
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* Right History Column: Inspector details */}
                  <div>
                    {inspectLog ? (
                      <div className="card">
                        <h3>🔎 Registry Record Inspector</h3>
                        <div style={{ display: 'flex', gap: '1.5rem', marginTop: '1.5rem', marginBottom: '1.5rem' }}>
                          <img 
                            src={inspectLog.thumbnail} 
                            alt="Inspect Preview" 
                            style={{ width: '150px', height: '150px', objectFit: 'cover', borderRadius: '8px', border: '1px solid var(--border-color)' }} 
                          />
                          <div>
                            <p style={{ fontWeight: 600 }}>{inspectLog.filename}</p>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-slate)' }}>Captured on: {inspectLog.timestamp}</p>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-slate)' }}>Engine: {inspectLog.model}</p>
                            
                            <div style={{ marginTop: '1rem' }}>
                              <span style={{ 
                                display: 'inline-block',
                                padding: '0.25rem 1rem', 
                                borderRadius: '4px', 
                                fontWeight: 700,
                                fontSize: '1.25rem',
                                color: inspectLog.verdict === 'AI' ? 'var(--accent-ai)' : 'var(--accent-real)',
                                backgroundColor: inspectLog.verdict === 'AI' ? 'var(--accent-ai-bg)' : 'var(--accent-real-bg)'
                              }}>
                                Verdict: {inspectLog.verdict_label || (inspectLog.verdict === 'AI' ? 'Likely AI-Generated' : 'Likely Real')}
                              </span>
                              {inspectLog.summary && (
                                <p style={{ marginTop: '0.75rem', fontSize: '0.88rem', color: 'var(--text-slate)', lineHeight: '1.4' }}>
                                  {inspectLog.summary}
                                </p>
                              )}
                            </div>
                          </div>
                        </div>

                        <div className="meter-section">
                          <div className="meter-header">
                            <span>AI Confidence Score</span>
                            <span>{(inspectLog.fake_score*100).toFixed(1)}%</span>
                          </div>
                          <div className="meter-track">
                            <div className="meter-fill ai" style={{ width: `${inspectLog.fake_score*100}%` }}></div>
                          </div>
                        </div>

                        <div className="meter-section" style={{ marginTop: '0.75rem' }}>
                          <div className="meter-header">
                            <span>Real Confidence Score</span>
                            <span style={{ color: 'var(--accent-real)' }}>{(inspectLog.real_score*100).toFixed(1)}%</span>
                          </div>
                          <div className="meter-track">
                            <div className="meter-fill real" style={{ width: `${inspectLog.real_score*100}%` }}></div>
                          </div>
                        </div>

                        {inspectLog.diagnostic_report?.contradiction_note && (
                          <div className="contradiction-notice" style={{ marginTop: '1.25rem' }}>
                            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                              <Info size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
                              <span>{inspectLog.diagnostic_report.contradiction_note}</span>
                            </div>
                          </div>
                        )}

                        <div style={{ marginTop: '1.5rem' }}>
                          <h4>Why did we reach this verdict?</h4>
                          <div className="reasons-list" style={{ marginTop: '0.75rem' }}>
                            {inspectLog.diagnostic_report?.evidence_cards ? (
                              inspectLog.diagnostic_report.evidence_cards.map((card, i) => (
                                <div key={i} className={`reason-item-card ${card.status_type || 'neutral'}`}>
                                  <span className="reason-icon">{card.icon}</span>
                                  <div className="reason-content">
                                    <div className="evidence-header">
                                      <strong className="reason-label" style={{ marginBottom: 0 }}>{card.title}</strong>
                                      <span className={`status-badge ${card.status_type || 'neutral'}`}>
                                        {card.status}
                                      </span>
                                    </div>
                                    <p className="reason-detail">{card.explanation}</p>
                                  </div>
                                </div>
                              ))
                            ) : (
                              inspectLog.reasons?.map((r, i) => {
                                const isObj = typeof r === 'object' && r !== null;
                                const type = isObj ? (r.type || 'neutral') : 'neutral';
                                const icon = isObj ? r.icon : 'ℹ️';
                                const label = isObj ? r.label : 'Diagnostic Audit';
                                const detail = isObj ? r.detail : String(r).replace(/\*\*/g, '');
                                
                                return (
                                  <div key={i} className={`reason-item-card ${type}`}>
                                    <span className="reason-icon">{icon}</span>
                                    <div className="reason-content">
                                      <strong className="reason-label">{label}</strong>
                                      <p className="reason-detail">{detail}</p>
                                    </div>
                                  </div>
                                );
                              })
                            )}
                          </div>
                        </div>

                        {inspectLog.diagnostic_report?.technical_details && (
                          <div>
                            <button
                              type="button"
                              className="tech-toggle-btn"
                              onClick={() => setShowHistoryTechDetails(prev => !prev)}
                            >
                              <span>View Technical Details</span>
                              {showHistoryTechDetails ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                            </button>

                            {showHistoryTechDetails && (
                              <div className="tech-drawer">
                                <div className="tech-grid">
                                  <div className="tech-metric-card">
                                    <span className="tech-metric-title">Model</span>
                                    <span className="tech-metric-value" style={{ fontSize: '0.92rem' }}>
                                      {inspectLog.diagnostic_report.technical_details.model_name || 'SigLIP Classifier'}
                                    </span>
                                    <span className="tech-metric-sub">
                                      AI: {(inspectLog.fake_score * 100).toFixed(1)}% | Real: {(inspectLog.real_score * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                  <div className="tech-metric-card">
                                    <span className="tech-metric-title">Error Level Analysis</span>
                                    <span className="tech-metric-value">
                                      {inspectLog.diagnostic_report.technical_details.ela_std}
                                    </span>
                                    <span className="tech-metric-sub">
                                      Threshold: &gt; {inspectLog.diagnostic_report.technical_details.ela_threshold} std
                                    </span>
                                  </div>
                                  <div className="tech-metric-card">
                                    <span className="tech-metric-title">FFT Frequency</span>
                                    <span className="tech-metric-value">
                                      {inspectLog.diagnostic_report.technical_details.fft_std}
                                    </span>
                                    <span className="tech-metric-sub">
                                      Threshold: &gt; {inspectLog.diagnostic_report.technical_details.fft_threshold} std
                                    </span>
                                  </div>
                                  <div className="tech-metric-card">
                                    <span className="tech-metric-title">Metadata Audit</span>
                                    <span className="tech-metric-value">
                                      {inspectLog.diagnostic_report.technical_details.has_exif ? `${inspectLog.diagnostic_report.technical_details.metadata_count} tags` : 'No EXIF'}
                                    </span>
                                    <span className="tech-metric-sub">
                                      {inspectLog.diagnostic_report.technical_details.ai_signatures?.length > 0 
                                        ? `AI Sigs: ${inspectLog.diagnostic_report.technical_details.ai_signatures.join(', ')}`
                                        : 'No direct AI tags'}
                                    </span>
                                  </div>
                                </div>

                                {inspectLog.diagnostic_report.technical_details.metadata_tags && Object.keys(inspectLog.diagnostic_report.technical_details.metadata_tags).length > 0 && (
                                  <div style={{ marginTop: '0.75rem', maxHeight: '180px', overflowY: 'auto' }}>
                                    <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-slate)' }}>Raw Metadata Headers:</span>
                                    <table className="exif-table" style={{ marginTop: '0.35rem', fontSize: '0.8rem' }}>
                                      <thead>
                                        <tr>
                                          <th style={{ padding: '0.4rem 0.6rem' }}>Tag</th>
                                          <th style={{ padding: '0.4rem 0.6rem' }}>Value</th>
                                        </tr>
                                      </thead>
                                      <tbody>
                                        {Object.entries(inspectLog.diagnostic_report.technical_details.metadata_tags).map(([k, v]) => (
                                          <tr key={k}>
                                            <td style={{ padding: '0.4rem 0.6rem', fontWeight: 600 }}>{k}</td>
                                            <td style={{ padding: '0.4rem 0.6rem' }}>{String(v)}</td>
                                          </tr>
                                        ))}
                                      </tbody>
                                    </table>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-slate)' }}>
                        <p>Select a registry entry from the list to inspect details.</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* -------------------------------------------------------------
            PAGE 4: BATCH
           ------------------------------------------------------------- */}
        {page === 'batch' && (
          <div>
            <div className="page-header">
              <h1>📁 Batch Processing Dashboard</h1>
              <p>Upload a folder or multiple files to evaluate classifications in bulk.</p>
            </div>

            <div className="card">
              <h3 style={{ marginBottom: '1rem' }}>📁 Bulk Upload Dropzone</h3>
              <label className="dropzone">
                <UploadCloud size={48} style={{ color: 'var(--primary-purple)' }} />
                <div>
                  <p>Select multiple images to batch process</p>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-slate)' }}>Supports multiple PNG, JPG, JPEG</span>
                </div>
                <input type="file" multiple onChange={handleBatchUpload} style={{ display: 'none' }} accept="image/*" />
              </label>
            </div>

            {batchResults.length > 0 && (
              <div>
                <div className="card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                    <h3>📋 Batch Diagnostic Summary</h3>
                    {!batching && (
                      <button className="btn-purple" onClick={handleExportCSV}>
                        <Download size={16} style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
                        Export Results (CSV)
                      </button>
                    )}
                  </div>

                  {batching && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem' }}>
                      <div className="loader"></div>
                      <span style={{ fontWeight: 600, color: 'var(--primary-purple)' }}>Bulk evaluating files...</span>
                    </div>
                  )}

                  <div className="stats-grid">
                    <div className="stat-card">
                      <span className="stat-label">Total Files</span>
                      <span className="stat-value">{batchResults.length}</span>
                    </div>
                    <div className="stat-card">
                      <span className="stat-label">AI Flagged</span>
                      <span className="stat-value" style={{ color: 'var(--accent-ai)' }}>
                        {batchResults.filter(x => x.verdict === 'AI').length}
                      </span>
                    </div>
                    <div className="stat-card">
                      <span className="stat-label">Real Flagged</span>
                      <span className="stat-value" style={{ color: 'var(--accent-real)' }}>
                        {batchResults.filter(x => x.verdict === 'REAL').length}
                      </span>
                    </div>
                    <div className="stat-card">
                      <span className="stat-label">AI Ratio</span>
                      <span className="stat-value">
                        {((batchResults.filter(x => x.verdict === 'AI').length / batchResults.length) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>

                  <h4>Batch Spreadsheet:</h4>
                  <table className="exif-table" style={{ marginTop: '0.5rem' }}>
                    <thead>
                      <tr>
                        <th>Filename</th>
                        <th>Verdict</th>
                        <th>AI Confidence</th>
                        <th>Real Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {batchResults.map((item) => (
                        <tr key={item.id}>
                          <td style={{ fontWeight: 600 }}>{item.filename}</td>
                          <td style={{ 
                            fontWeight: 700, 
                            color: item.verdict === 'AI' ? 'var(--accent-ai)' : (item.verdict === 'REAL' ? 'var(--accent-real)' : '#6b7280') 
                          }}>
                            {item.verdict_label || (item.verdict === 'AI' ? 'Likely AI-Generated' : (item.verdict === 'REAL' ? 'Likely Real' : item.verdict))}
                          </td>
                          <td>{(item.fake_score*100).toFixed(1)}%</td>
                          <td>{(item.real_score*100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="card">
                  <h3>🖼️ Grid Results Gallery</h3>
                  <p style={{ color: 'var(--text-slate)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                    Visual gallery results. Green border signifies Likely Real, Red border signifies Likely AI-Generated.
                  </p>

                  <div className="batch-grid">
                    {batchResults.map((item) => (
                      <div 
                        key={item.id} 
                        className="batch-thumb-card"
                        style={{ borderColor: item.verdict === 'AI' ? 'var(--accent-ai)' : (item.verdict === 'REAL' ? 'var(--accent-real)' : '#e2e8f0') }}
                      >
                        <span className={`batch-badge ${item.verdict === 'AI' ? 'ai' : 'real'}`}>
                          {item.verdict_label || (item.verdict === 'AI' ? 'Likely AI-Generated' : (item.verdict === 'REAL' ? 'Likely Real' : item.verdict))}
                        </span>
                        <p style={{ fontSize: '0.75rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontWeight: 600 }}>
                          {item.filename}
                        </p>
                        <img src={item.thumbnail} alt={item.filename} />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="footer">
          AI Image Spotter v2.0.0 | Decoupled React + FastAPI Edition | Powered by SigLIP Transformer
        </div>
      </div>
    </div>
  );
}
