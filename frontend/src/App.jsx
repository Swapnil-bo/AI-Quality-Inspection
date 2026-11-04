import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import "./App.css";

// Change this when deploying (Phase 6) -- points at the local
// FastAPI dev server for now.
const API_URL = "http://127.0.0.1:8000";

function StatusDot({ status }) {
  const labels = {
    checking: "Checking\u2026",
    ready: "Model Ready",
    offline: "Model Offline",
  };
  return (
    <span className={`status-dot status-dot--${status}`}>
      <span className="status-dot__light" />
      {labels[status]}
    </span>
  );
}

function ScoreGauge({ score, threshold, label }) {
  const reduceMotion = useReducedMotion();
  const isDefect = label === "defect";

  // Zoom the visible range around the two values so the gap between
  // score and threshold is always legible, regardless of the raw
  // scale PatchCore's distances happen to sit at.
  const lo = Math.min(score, threshold);
  const hi = Math.max(score, threshold);
  const pad = Math.max((hi - lo) * 0.6, 0.05);
  const min = Math.max(lo - pad, 0);
  const max = hi + pad;

  const toPct = (v) => ((v - min) / (max - min)) * 100;
  const scorePct = toPct(score);
  const thresholdPct = toPct(threshold);

  return (
    <div className="tag-panel__gauge">
      <div className="tag-panel__gauge-labels">
        <span>Score</span>
        <span>Threshold</span>
      </div>
      <div className="tag-panel__gauge-track">
        <motion.div
          className={`tag-panel__gauge-fill tag-panel__gauge-fill--${isDefect ? "defect" : "pass"}`}
          initial={{ width: 0 }}
          animate={{ width: `${scorePct}%` }}
          transition={{ duration: reduceMotion ? 0 : 0.6, ease: "easeOut", delay: reduceMotion ? 0 : 0.15 }}
        />
        <div className="tag-panel__gauge-tick" style={{ left: `${thresholdPct}%` }} />
        <span className="tag-panel__gauge-tick-label" style={{ left: `${thresholdPct}%` }}>
          {threshold.toFixed(3)}
        </span>
      </div>
    </div>
  );
}

function TagPanel({ status, result, error }) {
  const reduceMotion = useReducedMotion();

  return (
    <div className="tag-panel">
      <div className="tag-panel__punch" />
      <p className="tag-panel__eyebrow">Inspection Tag</p>

      {status === "idle" && (
        <div className="tag-panel__idle">
          <svg className="tag-panel__idle-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4">
            <path d="M9 12l2 2 4-4" />
            <circle cx="12" cy="12" r="9" />
          </svg>
          <p className="tag-panel__idle-text">Load a part on the left to run inspection. Results appear here.</p>
        </div>
      )}

      {status === "loading" && (
        <div className="tag-panel__idle">
          <p className="tag-panel__idle-text">Scanning for anomalies…</p>
        </div>
      )}

      {status === "error" && (
        <div className="tag-panel__error">
          <p className="tag-panel__error-title">Inspection failed</p>
          <p className="tag-panel__error-detail">{error}</p>
        </div>
      )}

      {status === "done" && result && (
        <>
          <div className="tag-panel__verdict-wrap">
            <motion.div
              className={`tag-panel__verdict tag-panel__verdict--${result.label === "defect" ? "defect" : "pass"}`}
              initial={reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 1.5, rotate: 6 }}
              animate={reduceMotion ? { opacity: 1 } : { opacity: 1, scale: 1, rotate: -3 }}
              transition={{ type: "spring", stiffness: 300, damping: 16 }}
            >
              {result.label === "defect" ? "Defect" : "Pass"}
            </motion.div>
          </div>

          <ScoreGauge score={result.score} threshold={result.threshold} label={result.label} />

          <div className="tag-panel__meta">
            <div className="tag-panel__meta-row">
              <span className="tag-panel__meta-key">Category</span>
              <span className="tag-panel__meta-value">{result.category}</span>
            </div>
            <div className="tag-panel__meta-row">
              <span className="tag-panel__meta-key">Raw score</span>
              <span className="tag-panel__meta-value">{result.score.toFixed(5)}</span>
            </div>
            <div className="tag-panel__meta-row">
              <span className="tag-panel__meta-key">Threshold</span>
              <span className="tag-panel__meta-value">{result.threshold.toFixed(5)}</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default function App() {
  const [apiStatus, setApiStatus] = useState("checking");
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [previewUrl, setPreviewUrl] = useState(null);
  const [overlayUrl, setOverlayUrl] = useState(null);
  const [fileName, setFileName] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((res) => (res.ok ? setApiStatus("ready") : setApiStatus("offline")))
      .catch(() => setApiStatus("offline"));
  }, []);

  const runInspection = useCallback(async (file) => {
    setFileName(file.name);
    setPreviewUrl(URL.createObjectURL(file));
    setOverlayUrl(null);
    setStatus("loading");
    setError("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail || `Server returned ${res.status}`);
      }
      const data = await res.json();
      setOverlayUrl(`data:image/png;base64,${data.overlay_base64}`);
      setResult(data);
      setStatus("done");
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Can't reach the inspection API. Check that it's running on :8000."
          : err.message
      );
      setStatus("error");
    }
  }, []);

  const handleFile = useCallback(
    (fileList) => {
      const file = fileList?.[0];
      if (!file) return;
      if (!file.type.startsWith("image/")) {
        setError("That file isn't an image. Choose a PNG or JPEG of the part.");
        setStatus("error");
        return;
      }
      runInspection(file);
    },
    [runInspection]
  );

  const reset = () => {
    setStatus("idle");
    setPreviewUrl(null);
    setOverlayUrl(null);
    setResult(null);
    setError("");
    setFileName("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const displayImage = overlayUrl || previewUrl;

  return (
    <div className="app">
      <header className="app__header">
        <div className="app__brand">
          <StatusDot status={apiStatus} />
        </div>
        <p className="app__wordmark">
          QC <span>//</span> TRANSISTOR INSPECTION
        </p>
        <a
          className="app__repo-link"
          href="https://github.com/Swapnil-bo/AI-Quality-Inspection"
          target="_blank"
          rel="noreferrer"
        >
          View source ↗
        </a>
      </header>

      <main className="app__main">
        <section className="scan-stage">
          <p className="scan-stage__eyebrow">01 — Scan Stage</p>

          {!displayImage ? (
            <div
              className={`scan-stage__frame scan-stage__frame--drop ${dragOver ? "scan-stage__frame--drag-over" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                handleFile(e.dataTransfer.files);
              }}
            >
              <div className="scan-stage__drop-content">
                <svg className="scan-stage__drop-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.3">
                  <path d="M12 16V4M12 4l-4 4M12 4l4 4" />
                  <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
                </svg>
                <p className="scan-stage__drop-title">Drop a component image, or click to browse</p>
                <p className="scan-stage__drop-sub">PNG / JPEG · trained on: transistor</p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="scan-stage__hidden-input"
                onChange={(e) => handleFile(e.target.files)}
              />
            </div>
          ) : (
            <div className="scan-stage__frame">
              <img src={displayImage} alt="Component under inspection" className="scan-stage__image" />
              {status === "loading" && (
                <>
                  <div className="scan-stage__beam" />
                  <span className="scan-stage__loading-label">Scanning…</span>
                </>
              )}
            </div>
          )}

          {displayImage && (
            <div className="scan-stage__actions">
              <span className="scan-stage__filename">{fileName}</span>
              <button className="scan-stage__reset" onClick={reset}>
                Scan another part
              </button>
            </div>
          )}
        </section>

        <section>
          <p className="scan-stage__eyebrow" style={{ marginBottom: 16 }}>
            02 — Result
          </p>
          <AnimatePresence mode="wait">
            <motion.div
              key={status + (result ? result.label : "")}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <TagPanel status={status} result={result} error={error} />
            </motion.div>
          </AnimatePresence>
        </section>
      </main>

      <footer className="app__footer">
        PatchCore · ResNet18 feature memory bank · F1 0.884 on the official MVTec AD test split
      </footer>
    </div>
  );
}