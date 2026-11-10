import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import "./App.css";

// Local dev default. For a deployed build, set VITE_API_URL (in
// frontend/.env or the host's environment variables) instead of
// editing this file.
const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
const SAMPLE_BASE = `${import.meta.env.BASE_URL}samples/`;

const MAX_HISTORY = 12;
const LINE_DELAY_MS = 700;
const HEALTH_POLL_MS = 8000;
const OPACITY_DEFECT = 0.5; // default heatmap strength on a DEFECT result
const OPACITY_GOOD = 0.2; // fainter on PASS so the overlay doesn't contradict the verdict

// One 000.png per MVTec transistor test folder, copied into
// frontend/public/samples/ (see the copy command in the instructions).
const SAMPLES = [
  { id: "good", label: "Good", expected: "good" },
  { id: "bent_lead", label: "Bent lead", expected: "defect" },
  { id: "cut_lead", label: "Cut lead", expected: "defect" },
  { id: "damaged_case", label: "Damaged case", expected: "defect" },
  { id: "misplaced", label: "Misplaced", expected: "defect" },
];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const toDataUrl = (b64) => (b64 ? `data:image/png;base64,${b64}` : null);
const defaultOpacity = (label) => (label === "defect" ? OPACITY_DEFECT : OPACITY_GOOD);

/* ------------------------------------------------------------------ */
/* small components                                                    */
/* ------------------------------------------------------------------ */

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
  const tone = label === "defect" ? "defect" : "pass";

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
  const transition = {
    duration: reduceMotion ? 0 : 0.6,
    ease: "easeOut",
    delay: reduceMotion ? 0 : 0.15,
  };

  return (
    <div className="tag-panel__gauge">
      <div className="tag-panel__gauge-track">
        <motion.div
          className={`tag-panel__gauge-fill tag-panel__gauge-fill--${tone}`}
          initial={{ width: 0 }}
          animate={{ width: `${scorePct}%` }}
          transition={transition}
        />
        <div className="tag-panel__gauge-tick" style={{ left: `${thresholdPct}%` }} />

        {/* each marker is labelled where it sits, so nothing is ambiguous */}
        <motion.span
          className={`tag-panel__gauge-flag tag-panel__gauge-flag--${tone}`}
          initial={{ left: "0%" }}
          animate={{ left: `${scorePct}%` }}
          transition={transition}
        >
          score {score.toFixed(3)}
        </motion.span>
        <span className="tag-panel__gauge-tick-label" style={{ left: `${thresholdPct}%` }}>
          threshold {threshold.toFixed(3)}
        </span>
      </div>
    </div>
  );
}

function TagPanel({ status, result, latencyMs, error }) {
  const reduceMotion = useReducedMotion();
  const isDefect = result?.label === "defect";

  const margin = result ? result.score - result.threshold : 0;
  const marginPct = result ? (margin / result.threshold) * 100 : 0;
  const sign = margin >= 0 ? "+" : "\u2212";

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
          <p className="tag-panel__idle-text">
            Load a part on the left, pick a sample, or run the line. Results appear here.
          </p>
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
              className={`tag-panel__verdict tag-panel__verdict--${isDefect ? "defect" : "pass"}`}
              initial={reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 1.5, rotate: 6 }}
              animate={reduceMotion ? { opacity: 1 } : { opacity: 1, scale: 1, rotate: -3 }}
              transition={{ type: "spring", stiffness: 300, damping: 16 }}
            >
              {isDefect ? "Defect" : "Pass"}
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
            <div className="tag-panel__meta-row">
              <span className="tag-panel__meta-key">Margin</span>
              <span className={`tag-panel__meta-value tag-panel__meta-value--${isDefect ? "defect" : "pass"}`}>
                {sign}
                {Math.abs(margin).toFixed(3)} ({sign}
                {Math.abs(marginPct).toFixed(1)}%)
              </span>
            </div>
            {latencyMs != null && (
              <div className="tag-panel__meta-row">
                <span className="tag-panel__meta-key">Round trip</span>
                <span className="tag-panel__meta-value">{latencyMs} ms</span>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function ViewControls({ mode, setMode, opacity, setOpacity }) {
  const modes = [
    ["original", "Original"],
    ["overlay", "Overlay"],
    ["heatmap", "Heatmap"],
  ];
  return (
    <div className="view-controls">
      <div className="view-controls__seg" role="group" aria-label="Image view">
        {modes.map(([key, text]) => (
          <button
            key={key}
            type="button"
            aria-pressed={mode === key}
            className={`view-controls__btn ${mode === key ? "view-controls__btn--active" : ""}`}
            onClick={() => setMode(key)}
          >
            {text}
          </button>
        ))}
      </div>
      {mode === "overlay" && (
        <label className="view-controls__slider">
          <span>Opacity</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
          />
          <span className="view-controls__pct">{Math.round(opacity * 100)}%</span>
        </label>
      )}
    </div>
  );
}

function SampleRow({ samples, disabled, onPick, onMissing }) {
  if (samples.length === 0) return null;
  return (
    <div className="samples">
      <p className="samples__label">Try a sample</p>
      <div className="samples__grid">
        {samples.map((s) => (
          <button
            key={s.id}
            type="button"
            className="samples__item"
            disabled={disabled}
            onClick={() => onPick(s)}
            title={`Inspect sample: ${s.label}`}
          >
            <img src={`${SAMPLE_BASE}${s.id}.png`} alt="" loading="lazy" onError={() => onMissing(s.id)} />
            <span>{s.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function LineLog({ history, tally, activeId, running, busy, canRun, onSelect, onRun, onStop, onClear }) {
  const total = tally.pass + tally.defect;
  return (
    <div className="line-log">
      <p className="scan-stage__eyebrow">03 — Line Log</p>
      <div className="line-log__panel">
        <div className="line-log__stats">
          <div className="line-log__stat">
            <span className="line-log__stat-num">{total}</span>
            <span className="line-log__stat-key">Scanned</span>
          </div>
          <div className="line-log__stat line-log__stat--pass">
            <span className="line-log__stat-num">{tally.pass}</span>
            <span className="line-log__stat-key">Pass</span>
          </div>
          <div className="line-log__stat line-log__stat--defect">
            <span className="line-log__stat-num">{tally.defect}</span>
            <span className="line-log__stat-key">Defect</span>
          </div>
        </div>

        <div className="line-log__actions">
          {running ? (
            <button type="button" className="line-log__btn line-log__btn--stop" onClick={onStop}>
              ■ Stop line
            </button>
          ) : (
            <button
              type="button"
              className="line-log__btn line-log__btn--run"
              onClick={onRun}
              disabled={!canRun || busy}
              title={canRun ? "Feed the sample parts through the model one at a time" : "Add images to frontend/public/samples first"}
            >
              ▶ Run line simulation
            </button>
          )}
          {total > 0 && !running && (
            <button type="button" className="line-log__btn" onClick={onClear}>
              Clear
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <p className="line-log__empty">No parts scanned yet.</p>
        ) : (
          <ul className="line-log__list">
            {history.map((entry) => {
              const isDefect = entry.result.label === "defect";
              const check = entry.expected ? (entry.expected === entry.result.label ? "match" : "miss") : null;
              return (
                <li key={entry.id}>
                  <button
                    type="button"
                    className={`line-log__row ${activeId === entry.id ? "line-log__row--active" : ""}`}
                    onClick={() => onSelect(entry)}
                  >
                    <img className="line-log__thumb" src={entry.originalUrl} alt="" />
                    <span className="line-log__name">{entry.name}</span>
                    <span className={`line-log__chip line-log__chip--${isDefect ? "defect" : "pass"}`}>
                      {isDefect ? "Defect" : "Pass"}
                    </span>
                    <span className="line-log__score">{entry.result.score.toFixed(3)}</span>
                    {check && (
                      <span
                        className={`line-log__check line-log__check--${check}`}
                        title={check === "match" ? "Matches the dataset label" : "Differs from the dataset label"}
                      >
                        {check === "match" ? "✓" : "✗"}
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* app                                                                 */
/* ------------------------------------------------------------------ */

export default function App() {
  const [apiStatus, setApiStatus] = useState("checking");
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [current, setCurrent] = useState(null);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [viewMode, setViewMode] = useState("overlay"); // original | overlay | heatmap
  const [heatOpacity, setHeatOpacity] = useState(OPACITY_DEFECT);
  const [history, setHistory] = useState([]);
  const [tally, setTally] = useState({ pass: 0, defect: 0 });
  const [lineRunning, setLineRunning] = useState(false);
  const [missingSamples, setMissingSamples] = useState({});

  const fileInputRef = useRef(null);
  const requestIdRef = useRef(0); // lets a reset cancel an in-flight request
  const entryIdRef = useRef(0);
  const stopLineRef = useRef(false);
  const prevHistoryRef = useRef([]);
  const currentUrlRef = useRef(null);

  const busy = status === "loading" || lineRunning;

  // Poll /health so the status dot recovers if the backend starts
  // after the page loads (or drops while it's open).
  useEffect(() => {
    let cancelled = false;
    const check = () =>
      fetch(`${API_URL}/health`)
        .then((res) => {
          if (!cancelled) setApiStatus(res.ok ? "ready" : "offline");
        })
        .catch(() => {
          if (!cancelled) setApiStatus("offline");
        });
    check();
    const id = setInterval(check, HEALTH_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // Free the blob URLs of scans that fall out of the log, unless
  // that scan is the one currently on the stage.
  useEffect(() => {
    currentUrlRef.current = current?.originalUrl ?? null;
  }, [current]);

  useEffect(() => {
    const keep = new Set(history.map((e) => e.originalUrl));
    prevHistoryRef.current.forEach((e) => {
      if (!keep.has(e.originalUrl) && e.originalUrl !== currentUrlRef.current) {
        URL.revokeObjectURL(e.originalUrl);
      }
    });
    prevHistoryRef.current = history;
  }, [history]);

  const showEntry = useCallback((entry) => {
    setCurrent(entry);
    setStatus("done");
    setError("");
    setViewMode("overlay");
    setHeatOpacity(defaultOpacity(entry.result.label));
  }, []);

  const runInspection = useCallback(
    async (file, expected) => {
      const requestId = ++requestIdRef.current;
      const originalUrl = URL.createObjectURL(file);
      setCurrent({ id: null, name: file.name, originalUrl, heatmapUrl: null, overlayUrl: null, result: null });
      setStatus("loading");
      setError("");

      const formData = new FormData();
      formData.append("file", file);
      const startedAt = performance.now();

      try {
        const res = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
        if (!res.ok) {
          const detail = await res.json().catch(() => null);
          throw new Error(detail?.detail || `Server returned ${res.status}`);
        }
        const data = await res.json();

        if (requestId !== requestIdRef.current) {
          URL.revokeObjectURL(originalUrl);
          return null;
        }

        const entry = {
          id: ++entryIdRef.current,
          name: file.name,
          expected,
          originalUrl,
          heatmapUrl: toDataUrl(data.heatmap_base64),
          overlayUrl: toDataUrl(data.overlay_base64),
          latencyMs: Math.round(performance.now() - startedAt),
          result: data,
        };
        showEntry(entry);
        setHistory((prev) => [entry, ...prev].slice(0, MAX_HISTORY));
        setTally((t) => (data.label === "defect" ? { ...t, defect: t.defect + 1 } : { ...t, pass: t.pass + 1 }));
        return entry;
      } catch (err) {
        if (requestId !== requestIdRef.current) {
          URL.revokeObjectURL(originalUrl);
          return null;
        }
        setError(
          err.message === "Failed to fetch" ? `Can't reach the inspection API at ${API_URL}.` : err.message
        );
        setStatus("error");
        return null;
      }
    },
    [showEntry]
  );

  const handleFile = useCallback(
    (file) => {
      if (!file || busy) return;
      if (!file.type.startsWith("image/")) {
        setError("That file isn't an image. Choose a PNG or JPEG of the part.");
        setStatus("error");
        return;
      }
      runInspection(file);
    },
    [busy, runInspection]
  );

  // Paste an image straight from the clipboard.
  useEffect(() => {
    const onPaste = (event) => {
      const file = Array.from(event.clipboardData?.files ?? []).find((f) => f.type.startsWith("image/"));
      if (file) {
        event.preventDefault();
        handleFile(file);
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [handleFile]);

  const loadSample = useCallback(async (sample) => {
    try {
      const res = await fetch(`${SAMPLE_BASE}${sample.id}.png`);
      // Vite answers unknown paths with index.html, so check the type too.
      const type = res.headers.get("content-type") ?? "";
      if (!res.ok || !type.startsWith("image/")) return null;
      const blob = await res.blob();
      return new File([blob], `${sample.id}.png`, { type: blob.type || "image/png" });
    } catch {
      return null;
    }
  }, []);

  const pickSample = useCallback(
    async (sample) => {
      if (busy) return;
      const file = await loadSample(sample);
      if (!file) {
        setMissingSamples((m) => ({ ...m, [sample.id]: true }));
        return;
      }
      runInspection(file, sample.expected);
    },
    [busy, loadSample, runInspection]
  );

  const runLine = useCallback(async () => {
    if (busy) return;
    const queue = SAMPLES.filter((s) => !missingSamples[s.id]);
    if (queue.length === 0) return;

    stopLineRef.current = false;
    setLineRunning(true);
    for (const sample of queue) {
      if (stopLineRef.current) break;
      const file = await loadSample(sample);
      if (!file) {
        setMissingSamples((m) => ({ ...m, [sample.id]: true }));
        continue;
      }
      const entry = await runInspection(file, sample.expected);
      if (!entry) break; // API down, or the run was reset
      await sleep(LINE_DELAY_MS);
    }
    setLineRunning(false);
  }, [busy, missingSamples, loadSample, runInspection]);

  const stopLine = () => {
    stopLineRef.current = true;
  };

  const reset = useCallback(() => {
    requestIdRef.current += 1; // drop any response still in flight
    stopLineRef.current = true;
    setStatus("idle");
    setCurrent(null);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  const clearLog = () => {
    reset();
    setHistory([]);
    setTally({ pass: 0, defect: 0 });
  };

  const openPicker = () => fileInputRef.current?.click();

  const result = current?.result ?? null;
  const hasImage = Boolean(current);
  const hasHeatmap = Boolean(current?.heatmapUrl);
  // If the backend predates the separate heatmap field, fall back to
  // the pre-blended overlay image and hide the view controls.
  const baseSrc = hasHeatmap ? current.originalUrl : (current?.overlayUrl ?? current?.originalUrl);
  const layerOpacity = viewMode === "original" ? 0 : viewMode === "heatmap" ? 1 : heatOpacity;
  const frameTone = status === "done" && result ? (result.label === "defect" ? "defect" : "pass") : null;
  const availableSamples = SAMPLES.filter((s) => !missingSamples[s.id]);

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
        <section
          className="scan-stage"
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget)) setDragOver(false);
          }}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFile(e.dataTransfer.files?.[0]);
          }}
        >
          <p className="scan-stage__eyebrow">01 — Scan Stage</p>

          {/* always mounted, so a new image can be loaded at any time */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="scan-stage__hidden-input"
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              handleFile(file);
            }}
          />

          {!hasImage ? (
            <div
              className={`scan-stage__frame scan-stage__frame--drop ${dragOver ? "scan-stage__frame--drag-over" : ""}`}
              role="button"
              tabIndex={0}
              aria-label="Upload a component image"
              onClick={openPicker}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  openPicker();
                }
              }}
            >
              <div className="scan-stage__drop-content">
                <svg className="scan-stage__drop-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.3">
                  <path d="M12 16V4M12 4l-4 4M12 4l4 4" />
                  <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
                </svg>
                <p className="scan-stage__drop-title">Drop a component image, or click to browse</p>
                <p className="scan-stage__drop-sub">PNG / JPEG · paste works too · trained on: transistor</p>
              </div>
            </div>
          ) : (
            <div
              className={`scan-stage__frame ${frameTone ? `scan-stage__frame--${frameTone}` : ""} ${dragOver ? "scan-stage__frame--drag-over" : ""}`}
            >
              <img src={baseSrc} alt="Component under inspection" className="scan-stage__image" />
              {hasHeatmap && (
                <img
                  src={current.heatmapUrl}
                  alt="Anomaly heatmap"
                  className="scan-stage__image scan-stage__image--heat"
                  style={{ opacity: layerOpacity }}
                />
              )}
              <span className="corner corner--tl" />
              <span className="corner corner--tr" />
              <span className="corner corner--bl" />
              <span className="corner corner--br" />
              {status === "loading" && (
                <>
                  <div className="scan-stage__beam" />
                  <span className="scan-stage__loading-label">Scanning…</span>
                </>
              )}
            </div>
          )}

          {status === "done" && hasHeatmap && (
            <ViewControls mode={viewMode} setMode={setViewMode} opacity={heatOpacity} setOpacity={setHeatOpacity} />
          )}

          {hasImage && (
            <div className="scan-stage__actions">
              <span className="scan-stage__filename">{current.name}</span>
              <button className="scan-stage__reset" onClick={reset}>
                Scan another part
              </button>
            </div>
          )}

          <SampleRow
            samples={availableSamples}
            disabled={busy}
            onPick={pickSample}
            onMissing={(id) => setMissingSamples((m) => ({ ...m, [id]: true }))}
          />
        </section>

        <section className="side">
          <p className="scan-stage__eyebrow" style={{ marginBottom: 16 }}>
            02 — Result
          </p>
          <AnimatePresence mode="wait">
            <motion.div
              key={`${status}-${current?.id ?? ""}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <TagPanel status={status} result={result} latencyMs={current?.latencyMs ?? null} error={error} />
            </motion.div>
          </AnimatePresence>

          <LineLog
            history={history}
            tally={tally}
            activeId={current?.id ?? null}
            running={lineRunning}
            busy={busy}
            canRun={availableSamples.length > 0}
            onSelect={showEntry}
            onRun={runLine}
            onStop={stopLine}
            onClear={clearLog}
          />
        </section>
      </main>

      <footer className="app__footer">
        PatchCore · ResNet18 feature memory bank · F1 0.884 on the official MVTec AD test split
        <br />
        Sample images from the{" "}
        <a href="https://www.mvtec.com/company/research/datasets/mvtec-ad" target="_blank" rel="noreferrer">
          MVTec AD dataset
        </a>{" "}
        (CC BY-NC-SA 4.0)
      </footer>
    </div>
  );
}
