import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import "./App.css";

import { askClinicalQuestion, analyzeEcho, fetchHealth } from "./api";
import EchoResult from "./components/EchoResult";
import PendingModel from "./components/PendingModel";

const initialClinicalData = {
  age: "",
  sex: "",
  symptoms: "",
  bloodPressure: "",
  heartRate: "",
  diabetes: false,
  hypertension: false,
  smoking: false,
};

const modalityConfig = {
  echo: {
    label: "Echocardiography",
    short: "ECHO",
    description: "2D cardiac ultrasound",
    formats: "DICOM, NIfTI, PNG, JPEG",
    accept: ".png,.jpg,.jpeg,.dcm,.nii,.nii.gz",
    analyzed: true,
  },
  ccta: {
    label: "Coronary CT angiography",
    short: "CCTA",
    description: "Cardiac CT / coronary anatomy",
    formats: "DICOM, NIfTI, ZIP series",
    accept: ".dcm,.nii,.nii.gz,.zip,.png,.jpg,.jpeg",
    analyzed: false,
  },
  ecg: {
    label: "Electrocardiography",
    short: "ECG",
    description: "Waveform trace or report",
    formats: "PNG, JPEG, PDF, CSV",
    accept: ".png,.jpg,.jpeg,.pdf,.csv",
    analyzed: false,
  },
};

const navItems = [
  { id: "case", number: "01", label: "Patient case" },
  { id: "results", number: "02", label: "Analysis" },
  { id: "explainability", number: "03", label: "Explainability" },
  { id: "assistant", number: "04", label: "Case assistant" },
];

function App() {
  const [clinicalData, setClinicalData] = useState(initialClinicalData);
  const [files, setFiles] = useState({
    echo: null,
    ccta: null,
    ecg: null,
  });

  const [dragOver, setDragOver] = useState(null);

  const [activeSection, setActiveSection] = useState("case");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [activeResult, setActiveResult] = useState("overview");

  /* Real backend state */
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [echoResult, setEchoResult] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);

  /* Explicit image geometry. Defaults to no transform: the backend never
     guesses an orientation, and neither does this UI. */
  const [echoRotate, setEchoRotate] = useState(0);
  const [echoFlip, setEchoFlip] = useState(false);

  const [question, setQuestion] = useState("");
  const [conversation, setConversation] = useState([]);
  const [isAsking, setIsAsking] = useState(false);

  const echoInput = useRef(null);
  const cctaInput = useRef(null);
  const ecgInput = useRef(null);

  const inputRefs = {
    echo: echoInput,
    ccta: cctaInput,
    ecg: ecgInput,
  };

  const patientId = useMemo(
    () => `CV-${new Date().getFullYear()}-001`,
    []
  );

  /* ==========================================================
     BACKEND HEALTH

     The UI advertises capabilities based on what the backend
     actually reports, so it can never claim a model it lacks.
     ========================================================== */

  const loadHealth = useCallback(async () => {
    try {
      const data = await fetchHealth();
      setHealth(data);
      setHealthError(null);
    } catch (error) {
      setHealth(null);
      setHealthError(error.message);
    }
  }, []);

  useEffect(() => {
    loadHealth();
  }, [loadHealth]);

  const echoModelReady = Boolean(health?.modalities?.echo?.available);
  const medgemmaReady = Boolean(health?.models?.medgemma?.loaded);
  const backendOnline = Boolean(health);

  const uploadedCount = Object.values(files).filter(Boolean).length;

  const hasClinicalData = Object.values(clinicalData).some(
    (value) => value !== "" && value !== false
  );

  /* Analysis needs an echo image: it is the only trained imaging model. */
  const canAnalyze = Boolean(files.echo) && echoModelReady && !isAnalyzing;

  const updateClinical = (field, value) => {
    setClinicalData((previous) => ({
      ...previous,
      [field]: value,
    }));
  };

  const handleFile = (modality, file) => {
    if (!file) return;

    setFiles((previous) => ({
      ...previous,
      [modality]: file,
    }));

    if (modality === "echo") {
      setAnalysisComplete(false);
      setEchoResult(null);
      setAnalysisError(null);
      // A new image gets a clean slate: the previous file's rotation says
      // nothing about how this one was exported.
      setEchoRotate(0);
      setEchoFlip(false);
    }
  };

  const removeFile = (modality) => {
    setFiles((previous) => ({
      ...previous,
      [modality]: null,
    }));

    if (modality === "echo") {
      setAnalysisComplete(false);
      setEchoResult(null);
      setAnalysisError(null);
      setEchoRotate(0);
      setEchoFlip(false);
    }
  };

  const scrollToSection = useCallback((section) => {
    setActiveSection(section);

    const element = document.getElementById(section);

    if (element) {
      element.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }, []);

  /* ==========================================================
     REAL ANALYSIS

     Calls POST /api/analyze/echo and renders the model's actual
     output. There are no simulated results and no placeholder
     metrics anywhere in this flow.
     ========================================================== */

  const runAnalysis = useCallback(
    async (overrides = {}) => {
      if (!files.echo || !echoModelReady || isAnalyzing) return;

      const rotate = overrides.rotate ?? echoRotate;
      const flip = overrides.flip ?? echoFlip;

      setIsAnalyzing(true);
      setAnalysisError(null);
      setEchoResult(null);
      setAnalysisComplete(false);
      setEchoRotate(rotate);
      setEchoFlip(flip);

      scrollToSection("results");

      try {
        const result = await analyzeEcho(files.echo, { rotate, flip });

        setEchoResult(result);
        setAnalysisComplete(true);
        setActiveResult("echo");
      } catch (error) {
        setAnalysisError(error.message);
        // Refresh health so the UI reflects a backend that went away.
        loadHealth();
      } finally {
        setIsAnalyzing(false);
      }
    },
    [files.echo, echoModelReady, isAnalyzing, echoRotate, echoFlip, loadHealth]
  );

  /* Re-run the same image at a different rotation. Used by the orientation
     control in the echo result, since a display-oriented upload is a
     quarter turn away from the training distribution. */
  const reanalyzeWithOrientation = useCallback(
    (rotate, flip) => runAnalysis({ rotate, flip }),
    [runAnalysis]
  );

  /* ==========================================================
     CASE STATE FOR THE LANGUAGE MODEL

     Sent structured; the backend renders it into prompt text,
     including an explicit list of unavailable modalities so the
     model cannot invent CCTA or ECG findings.
     ========================================================== */

  const caseState = useMemo(
    () => ({
      case_id: patientId,
      clinical: clinicalData,
      echo: echoResult
        ? {
            analyzed: true,
            model: echoResult.model,
            structures: echoResult.structures,
            input: echoResult.input,
            // Both bear on how far the findings can be trusted, so the
            // backend needs them to caveat the prompt honestly.
            orientation: echoResult.orientation,
            quantification: echoResult.quantification,
          }
        : { analyzed: false },
      modalities_provided: {
        echo: Boolean(files.echo),
        ccta: Boolean(files.ccta),
        ecg: Boolean(files.ecg),
      },
    }),
    [patientId, clinicalData, echoResult, files]
  );

  const askQuestion = async (text) => {
    const trimmed = (text ?? question).trim();

    if (!trimmed || isAsking) return;

    setConversation((previous) => [
      ...previous,
      { role: "user", text: trimmed },
    ]);

    setQuestion("");
    setIsAsking(true);

    try {
      const data = await askClinicalQuestion(trimmed, caseState);

      setConversation((previous) => [
        ...previous,
        {
          role: "assistant",
          text:
            data.answer ||
            "No response was returned by the clinical language model.",
          model: data.model,
          device: data.device,
          contextUsed: data.context_used,
          contextPreview: data.context_preview,
        },
      ]);
    } catch (error) {
      setConversation((previous) => [
        ...previous,
        {
          role: "assistant",
          text: error.message,
          error: true,
        },
      ]);
    } finally {
      setIsAsking(false);
    }
  };

  const contextItems = [
    { label: "Clinical", active: hasClinicalData },
    { label: "Echo findings", active: Boolean(echoResult) },
    { label: "CCTA", active: false, unavailable: true },
    { label: "ECG", active: false, unavailable: true },
  ];

  return (
    <div className="cv-shell">
      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="cv-header">
        <div className="cv-brand">
          <div className="cv-brand-mark">CV</div>

          <div>
            <div className="cv-brand-name">CardioVision</div>

            <div className="cv-brand-subtitle">
              Multimodal cardiovascular intelligence
            </div>
          </div>
        </div>

        <div
          className={`cv-header-center ${backendOnline ? "" : "offline"}`}
          title={healthError || undefined}
        >
          <span className="cv-system-dot" />

          {backendOnline
            ? `Local inference · ${(health.device || "unknown").toUpperCase()} · ` +
              `echo ${echoModelReady ? "ready" : "unavailable"} · ` +
              `MedGemma ${medgemmaReady ? "ready" : "unavailable"}`
            : "Backend offline"}
        </div>

        <div className="cv-header-actions">
          <button
            className="cv-header-button"
            onClick={() => scrollToSection("case")}
          >
            New case
          </button>

          <div className="cv-user-avatar">MH</div>
        </div>
      </header>

      {/* ======================================================
          MOBILE NAV
      ====================================================== */}

      <nav className="cv-mobile-nav">
        {navItems.map((item) => (
          <button
            key={item.id}
            className={activeSection === item.id ? "active" : ""}
            onClick={() => scrollToSection(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <div className="cv-body">
        {/* ====================================================
            SIDEBAR
        ==================================================== */}

        <aside className="cv-sidebar">
          <div className="cv-sidebar-label">Workspace</div>

          {navItems.map((item) => (
            <button
              key={item.id}
              className={`cv-nav-item ${
                activeSection === item.id ? "active" : ""
              }`}
              onClick={() => scrollToSection(item.id)}
            >
              <span className="cv-nav-number">{item.number}</span>

              <span>{item.label}</span>
            </button>
          ))}

          <div className="cv-sidebar-spacer" />

          <div className="cv-local-card">
            <div className="cv-local-icon">◉</div>

            <div>
              <div className="cv-local-title">Local inference</div>

              <div className="cv-local-text">
                Patient data remains on this device.
              </div>
            </div>
          </div>
        </aside>

        {/* ====================================================
            MAIN
        ==================================================== */}

        <main className="cv-main">
          {/* ==================================================
              HERO
          ================================================== */}

          <section className="cv-hero">
            <div className="cv-eyebrow">Cardiovascular AI platform</div>

            <h1>Understand the heart through multimodal AI.</h1>

            <p>
              Echocardiography segmentation runs locally on a trained
              UNet++ model. CCTA, ECG and clinical risk models are still
              in development and are clearly marked as unavailable.
            </p>
          </section>

          {!backendOnline && (
            <div className="cv-banner error">
              <strong>Backend offline</strong>

              <span>{healthError}</span>

              <button className="cv-secondary-button" onClick={loadHealth}>
                Retry
              </button>
            </div>
          )}

          {backendOnline && !echoModelReady && (
            <div className="cv-banner warning">
              <strong>Echo model unavailable</strong>

              <span>
                {health?.modalities?.echo?.note ||
                  health?.models?.echo?.error ||
                  "The segmentation model did not load."}
              </span>

              <button className="cv-secondary-button" onClick={loadHealth}>
                Recheck
              </button>
            </div>
          )}

          {/* ==================================================
              01. PATIENT CASE
          ================================================== */}

          <section id="case" className="cv-section">
            <div className="cv-section-head">
              <div>
                <div className="cv-section-index">01 / Patient case</div>

                <h2>Build a clinical case</h2>

                <p>
                  Provide the available patient information and imaging
                  studies. Missing modalities can remain unavailable.
                </p>
              </div>

              <div className="cv-case-chip">
                <span>Case ID</span>
                <strong>{patientId}</strong>
              </div>
            </div>

            <div className="cv-case-grid">
              <ClinicalForm data={clinicalData} onChange={updateClinical} />

              <div className="cv-modality-panel">
                <div className="cv-panel-header">
                  <span className="cv-card-kicker">Imaging studies</span>

                  <h3>Modalities</h3>
                </div>

                <div className="cv-modality-list">
                  {Object.entries(modalityConfig).map(([key, modality]) => (
                    <ModalityRow
                      key={key}
                      modality={modality}
                      file={files[key]}
                      inputRef={inputRefs[key]}
                      isDragOver={dragOver === key}
                      onDragOver={(over) => setDragOver(over ? key : null)}
                      onUpload={(file) => handleFile(key, file)}
                      onRemove={() => removeFile(key)}
                    />
                  ))}
                </div>
              </div>
            </div>

            <div className="cv-analysis-bar">
              <div>
                <div className="cv-analysis-title">
                  {files.echo
                    ? "Ready to segment echocardiography"
                    : "An echo image is required to run analysis"}
                </div>

                <div className="cv-analysis-subtitle">
                  {files.echo
                    ? `${files.echo.name} · ${
                        hasClinicalData
                          ? "clinical data will be passed to the case assistant"
                          : "no clinical data entered"
                      }`
                    : "Echocardiography is the only trained imaging model. " +
                      "Clinical data alone can still be discussed in the case assistant."}
                </div>
              </div>

              <button
                className={`cv-primary-button ${isAnalyzing ? "loading" : ""}`}
                disabled={!canAnalyze}
                onClick={() => runAnalysis()}
              >
                {isAnalyzing ? (
                  <>
                    <span className="cv-button-spinner" />
                    Segmenting
                  </>
                ) : (
                  <>
                    Analyze echo
                    <span aria-hidden="true">→</span>
                  </>
                )}
              </button>
            </div>
          </section>

          {/* ==================================================
              02. RESULTS
          ================================================== */}

          <section id="results" className="cv-section">
            <div className="cv-section-head">
              <div>
                <div className="cv-section-index">02 / Analysis</div>

                <h2>Case intelligence</h2>

                <p>
                  Model output for every modality that has a trained model
                  behind it.
                </p>
              </div>

              <div
                className={`cv-status-chip ${
                  analysisComplete ? "complete" : ""
                }`}
              >
                <span />

                {analysisComplete ? "Analysis complete" : "Awaiting analysis"}
              </div>
            </div>

            {analysisError && (
              <div className="cv-banner error">
                <strong>Analysis failed</strong>

                <span>{analysisError}</span>
              </div>
            )}

            {!analysisComplete && !isAnalyzing && !analysisError && (
              <div className="cv-empty-analysis">
                <div className="cv-empty-icon">◌</div>

                <h3>No analysis generated yet</h3>

                <p>
                  Upload an echocardiography image above, then run the
                  segmentation model.
                </p>

                <button
                  className="cv-secondary-button"
                  onClick={() => scrollToSection("case")}
                >
                  Configure patient case
                </button>
              </div>
            )}

            {isAnalyzing && <AnalysisLoader />}

            {analysisComplete && (
              <div className="cv-results">
                <div className="cv-tabs" role="tablist">
                  {[
                    ["overview", "Overview", true],
                    ["echo", "Echo", true],
                    ["ccta", "CCTA", false],
                    ["ecg", "ECG", false],
                    ["clinical", "Clinical", false],
                  ].map(([id, label, available]) => (
                    <button
                      key={id}
                      role="tab"
                      aria-selected={activeResult === id}
                      className={`${activeResult === id ? "active" : ""} ${
                        available ? "" : "pending"
                      }`}
                      onClick={() => setActiveResult(id)}
                    >
                      {label}
                      {!available && <i className="cv-tab-dot" />}
                    </button>
                  ))}
                </div>

                {activeResult === "overview" && (
                  <OverviewResult
                    echoResult={echoResult}
                    files={files}
                    clinicalData={clinicalData}
                    hasClinicalData={hasClinicalData}
                  />
                )}

                {activeResult === "echo" &&
                  (echoResult ? (
                    <EchoResult
                      result={echoResult}
                      rotate={echoRotate}
                      flip={echoFlip}
                      onReorient={reanalyzeWithOrientation}
                      isAnalyzing={isAnalyzing}
                    />
                  ) : (
                    <div className="cv-empty-analysis">
                      <p>No echo image was analysed in this case.</p>
                    </div>
                  ))}

                {activeResult === "ccta" && (
                  <PendingModel
                    label="Coronary CT angiography"
                    file={files.ccta}
                    note="There is no trained CCTA model yet, so nothing is inferred from CT data."
                    requirement="notebooks/01_CCTA_Training.ipynb needs to be written and run, and the resulting weights saved to models/ccta/."
                  />
                )}

                {activeResult === "ecg" && (
                  <PendingModel
                    label="Electrocardiography"
                    file={files.ecg}
                    note="There is no ECG pipeline yet — no training notebook and no model."
                    requirement="An ECG training pipeline and a saved model in models/ecg/."
                  />
                )}

                {activeResult === "clinical" && (
                  <ClinicalResult clinicalData={clinicalData} />
                )}
              </div>
            )}
          </section>

          {/* ==================================================
              03. EXPLAINABILITY
          ================================================== */}

          {analysisComplete && echoResult && (
            <section id="explainability" className="cv-section">
              <div className="cv-section-head">
                <div>
                  <div className="cv-section-index">03 / Explainability</div>

                  <h2>See what the model responded to</h2>

                  <p>
                    {echoResult.explainability?.available
                      ? echoResult.explainability?.description
                      : "No attribution map was produced for this run, so " +
                        "there is nothing to show here."}
                  </p>
                </div>
              </div>

              {echoResult.explainability?.available ? (
                <>
                  <div className="cv-xai-grid">
                    <div className="cv-xai-image">
                      <div className="cv-echo-image-frame">
                        <img
                          src={echoResult.images?.saliency_overlay}
                          alt="Gradient saliency overlay"
                        />
                      </div>

                      <div className="cv-image-caption">
                        <strong>{echoResult.explainability?.method}</strong>

                        <span>
                          Target: {echoResult.explainability?.target_class}
                        </span>
                      </div>
                    </div>

                    <div className="cv-xai-image">
                      <div className="cv-echo-image-frame">
                        <img
                          src={echoResult.images?.combined}
                          alt="Segmentation and saliency combined"
                        />
                      </div>

                      <div className="cv-image-caption">
                        <strong>Segmentation + saliency</strong>

                        <span>
                          Predicted structures with attribution overlaid
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="cv-xai-note">
                    Saliency shows which input pixels the
                    left-ventricular-cavity probability was most sensitive
                    to. It is a property of the model, not independent
                    clinical evidence, and a bright region is not a finding.
                  </div>
                </>
              ) : (
                /* An all-zero gradient renders as a smooth, plausible-looking
                   heatmap. Showing it would be worse than showing nothing:
                   it is an explanation of a computation that never happened. */
                <div className="cv-xai-unavailable">
                  <strong>Attribution unavailable for this run</strong>

                  <p>
                    The gradient with respect to the input could not be
                    computed, so no saliency map exists. The segmentation
                    above is unaffected and remains valid — only the
                    explanation is missing.
                  </p>

                  {echoResult.notes?.length > 0 && (
                    <ul>
                      {echoResult.notes.map((note, index) => (
                        <li key={index}>{note}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </section>
          )}

          {/* ==================================================
              04. CASE ASSISTANT
          ================================================== */}

          <section id="assistant" className="cv-section cv-assistant-section">
            <div className="cv-section-head">
              <div>
                <div className="cv-section-index">04 / Case assistant</div>

                <h2>Ask about this case</h2>

                <p>
                  Query the local medical language model. When a case has
                  clinical data or echo findings, they are sent as context
                  and the model is told which modalities are unavailable.
                </p>
              </div>

              <div
                className={`cv-local-badge ${medgemmaReady ? "" : "offline"}`}
              >
                <span />
                {medgemmaReady
                  ? "Local medical model"
                  : "Language model unavailable"}
              </div>
            </div>

            <div className="cv-assistant">
              {/* ============================================
                  CASE CONTEXT
              ============================================ */}

              <div className="cv-assistant-context">
                <div className="cv-assistant-context-header">
                  <span className="cv-card-kicker">Case context</span>

                  <span className="cv-context-status">
                    {hasClinicalData || echoResult ? "Loaded" : "Empty"}
                  </span>
                </div>

                <div className="cv-context-items">
                  {contextItems.map((item) => (
                    <ContextItem
                      key={item.label}
                      label={item.label}
                      active={item.active}
                      unavailable={item.unavailable}
                    />
                  ))}
                </div>

                <div className="cv-context-hint">
                  Unavailable modalities are named explicitly in the prompt so
                  the model cannot invent findings for them.
                </div>
              </div>

              {/* ============================================
                  CHAT
              ============================================ */}

              <div className="cv-chat">
                {conversation.length === 0 ? (
                  <div className="cv-chat-empty">
                    <div className="cv-chat-icon">+</div>

                    <h3>Ask a clinical question</h3>

                    <p>
                      Ask about general cardiology, or about the findings in
                      this case.
                    </p>

                    <div className="cv-question-suggestions">
                      <button
                        onClick={() =>
                          askQuestion(
                            "What are the major risk factors for coronary artery disease?"
                          )
                        }
                        disabled={isAsking || !medgemmaReady}
                      >
                        CAD risk factors
                      </button>

                      <button
                        onClick={() =>
                          askQuestion(
                            "What do the segmented cardiac structures in this case show?"
                          )
                        }
                        disabled={isAsking || !medgemmaReady || !echoResult}
                      >
                        Explain these findings
                      </button>

                      <button
                        onClick={() =>
                          askQuestion(
                            "Based on the available data, what further information would be needed?"
                          )
                        }
                        disabled={isAsking || !medgemmaReady}
                      >
                        What is missing?
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="cv-conversation">
                    {conversation.map((message, index) => (
                      <ChatMessage key={index} message={message} />
                    ))}

                    {isAsking && (
                      <div className="cv-message assistant">
                        <div className="cv-message-role">CardioVision</div>

                        <div className="cv-message-text cv-thinking">
                          <span />
                          <span />
                          <span />
                          <em>MedGemma is generating...</em>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* ==========================================
                    INPUT
                ========================================== */}

                <div className="cv-chat-input">
                  <textarea
                    rows="1"
                    placeholder={
                      !medgemmaReady
                        ? "The language model is not loaded."
                        : isAsking
                        ? "MedGemma is generating a response..."
                        : "Ask a clinical question..."
                    }
                    value={question}
                    disabled={isAsking || !medgemmaReady}
                    onChange={(event) => setQuestion(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();

                        if (!isAsking) {
                          askQuestion();
                        }
                      }
                    }}
                  />

                  <button
                    onClick={() => askQuestion()}
                    disabled={!question.trim() || isAsking || !medgemmaReady}
                  >
                    {isAsking ? (
                      <>
                        <span className="cv-button-spinner" />
                        Asking
                      </>
                    ) : (
                      <>
                        Ask
                        <span aria-hidden="true">→</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </section>

          {/* ==================================================
              FOOTER
          ================================================== */}

          <footer className="cv-footer">
            <div>
              <strong>CardioVision AI</strong>

              <span>
                Multimodal cardiovascular imaging research platform
              </span>
            </div>

            <div>Research prototype · Local inference</div>
          </footer>
        </main>
      </div>
    </div>
  );
}

/* ============================================================
   CLINICAL FORM
   ============================================================ */

function ClinicalForm({ data, onChange }) {
  return (
    <div className="cv-clinical-card">
      <div className="cv-panel-header">
        <span className="cv-card-kicker">Clinical data</span>

        <h3>Patient information</h3>
      </div>

      <div className="cv-form">
        <div className="cv-form-row">
          <label>
            Age

            <input
              type="number"
              placeholder="e.g. 58"
              value={data.age}
              onChange={(event) => onChange("age", event.target.value)}
            />
          </label>

          <label>
            Sex

            <select
              value={data.sex}
              onChange={(event) => onChange("sex", event.target.value)}
            >
              <option value="">Select</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </label>
        </div>

        <label>
          Primary symptoms

          <textarea
            rows="3"
            placeholder="Chest pain, dyspnea, fatigue..."
            value={data.symptoms}
            onChange={(event) => onChange("symptoms", event.target.value)}
          />
        </label>

        <div className="cv-form-row">
          <label>
            Blood pressure

            <input
              type="text"
              placeholder="120/80 mmHg"
              value={data.bloodPressure}
              onChange={(event) =>
                onChange("bloodPressure", event.target.value)
              }
            />
          </label>

          <label>
            Heart rate

            <input
              type="number"
              placeholder="72 bpm"
              value={data.heartRate}
              onChange={(event) => onChange("heartRate", event.target.value)}
            />
          </label>
        </div>

        <div className="cv-checkbox-grid">
          {[
            ["diabetes", "Diabetes"],
            ["hypertension", "Hypertension"],
            ["smoking", "Smoking"],
          ].map(([field, label]) => (
            <label key={field} className="cv-checkbox">
              <input
                type="checkbox"
                checked={data[field]}
                onChange={(event) => onChange(field, event.target.checked)}
              />

              <span>{label}</span>
            </label>
          ))}
        </div>

        <div className="cv-form-note">
          There is no trained clinical risk model yet. These values are not
          scored — they are passed to the language model as case context.
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   MODALITY ROW
   ============================================================ */

function ModalityRow({
  modality,
  file,
  inputRef,
  isDragOver,
  onDragOver,
  onUpload,
  onRemove,
}) {
  const status = file ? "ready" : "empty";

  return (
    <div
      className={`cv-modality-row ${status} ${isDragOver ? "dragging" : ""} ${
        modality.analyzed ? "" : "unsupported"
      }`}
      onDragOver={(event) => {
        event.preventDefault();
        onDragOver(true);
      }}
      onDragLeave={() => onDragOver(false)}
      onDrop={(event) => {
        event.preventDefault();
        onDragOver(false);
        onUpload(event.dataTransfer.files?.[0]);
      }}
    >
      <div className="cv-modality-icon">{modality.short}</div>

      <div className="cv-modality-info">
        <div className="cv-modality-title">
          {modality.label}

          {!modality.analyzed && (
            <span className="cv-inline-badge">No model yet</span>
          )}
        </div>

        <div className="cv-modality-description">
          {file ? file.name : modality.description}
        </div>

        <div className="cv-modality-formats">
          {file
            ? `${(file.size / 1024 / 1024).toFixed(2)} MB`
            : modality.formats}
        </div>
      </div>

      <div className="cv-modality-actions">
        {file ? (
          <>
            <span className="cv-ready-badge">✓ Loaded</span>

            <button
              className="cv-ghost-button"
              onClick={() => inputRef.current?.click()}
            >
              Replace
            </button>

            <button className="cv-ghost-button danger" onClick={onRemove}>
              Remove
            </button>
          </>
        ) : (
          <button
            className="cv-ghost-button"
            onClick={() => inputRef.current?.click()}
          >
            Select file
          </button>
        )}

        <input
          ref={inputRef}
          type="file"
          accept={modality.accept}
          hidden
          onChange={(event) => onUpload(event.target.files?.[0])}
        />
      </div>
    </div>
  );
}

/* ============================================================
   ANALYSIS LOADER
   ============================================================ */

function AnalysisLoader() {
  return (
    <div className="cv-analysis-loader">
      <div className="cv-loader-spinner" />

      <div>
        <h3>Running echo segmentation</h3>

        <p>
          Preprocessing the frame and running the UNet++ model, then
          computing gradient saliency. First run also loads the weights.
        </p>
      </div>

      <div className="cv-loader-steps">
        <LoaderStep number="01" text="Decoding image" active />
        <LoaderStep number="02" text="Segmentation" active />
        <LoaderStep number="03" text="Quantification" active />
        <LoaderStep number="04" text="Explainability" active />
      </div>
    </div>
  );
}

function LoaderStep({ number, text, active }) {
  return (
    <div className={`cv-loader-step ${active ? "active" : ""}`}>
      <span>{number}</span>

      {text}
    </div>
  );
}

/* ============================================================
   OVERVIEW RESULT

   Summarises only what was actually computed. No aggregate
   "case confidence" is shown, because nothing in this system
   produces one.
   ============================================================ */

function OverviewResult({ echoResult, files, clinicalData, hasClinicalData }) {
  const structures = echoResult?.structures || [];
  const present = structures.filter((structure) => structure.present);
  const calibrated = Boolean(echoResult?.input?.has_spatial_calibration);

  // Derived from the model card rather than hardcoded, so adding a class
  // to the model cannot leave a stale denominator on screen.
  const foregroundCount = echoResult?.model?.num_classes
    ? echoResult.model.num_classes - 1
    : structures.length;

  return (
    <div className="cv-overview">
      <div className="cv-summary-card">
        <div className="cv-summary-main">
          <span className="cv-card-kicker">What was computed</span>

          <h3>
            {echoResult
              ? `Echocardiography segmented — ${present.length} of ${foregroundCount} structures identified`
              : "No imaging analysis in this case"}
          </h3>

          <p>
            {echoResult
              ? "The echo model outlined cardiac structures in the supplied " +
                "frame. This is anatomical segmentation, not a diagnosis or " +
                "a risk score."
              : "Upload an echocardiography image to run the only trained " +
                "imaging model."}
          </p>
        </div>

        {echoResult && (
          <div className="cv-summary-confidence">
            <span>Model test Dice</span>

            <strong>{echoResult.model?.metrics?.test_dice}</strong>

            <em>dataset-level, not per-case</em>
          </div>
        )}
      </div>

      <div className="cv-metric-grid">
        <Metric
          label="Echo segmentation"
          value={echoResult ? "Complete" : "Not run"}
          tone={echoResult ? "" : "muted"}
        />

        <Metric label="CCTA" value="No model" tone="muted" />

        <Metric label="ECG" value="No model" tone="muted" />

        <Metric label="Clinical risk score" value="No model" tone="muted" />
      </div>

      <div className="cv-findings-grid">
        <div className="cv-finding-card">
          <span className="cv-card-kicker">Segmentation findings</span>

          {structures.length === 0 && (
            <p className="cv-muted-text">No structures were segmented.</p>
          )}

          {structures.map((structure) => (
            <Finding
              key={structure.class_index}
              title={structure.name}
              value={
                structure.present
                  ? calibrated && structure.area_cm2
                    ? `${structure.area_cm2.toFixed(2)} cm²`
                    : `${structure.area_percent.toFixed(1)}% of field`
                  : "Not identified"
              }
              muted={!structure.present}
            />
          ))}
        </div>

        <div className="cv-finding-card cv-next-step">
          <span className="cv-card-kicker">Case inputs</span>

          <h3>What this case contains</h3>

          <ul className="cv-input-list">
            <li>
              Clinical data:{" "}
              {hasClinicalData
                ? `entered${clinicalData.age ? `, age ${clinicalData.age}` : ""}`
                : "none entered"}
            </li>

            <li>Echo image: {files.echo ? files.echo.name : "none"}</li>

            <li>
              CCTA: {files.ccta ? `${files.ccta.name} (not analysed)` : "none"}
            </li>

            <li>
              ECG: {files.ecg ? `${files.ecg.name} (not analysed)` : "none"}
            </li>
          </ul>

          <p>
            Review the segmentation and explainability output before drawing
            any clinical conclusion.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   CLINICAL RESULT
   ============================================================ */

function ClinicalResult({ clinicalData }) {
  const entries = [
    ["Age", clinicalData.age],
    ["Sex", clinicalData.sex],
    ["Blood pressure", clinicalData.bloodPressure],
    [
      "Heart rate",
      clinicalData.heartRate ? `${clinicalData.heartRate} bpm` : "",
    ],
  ];

  const riskFactors = [
    ["Diabetes", clinicalData.diabetes],
    ["Hypertension", clinicalData.hypertension],
    ["Smoking", clinicalData.smoking],
  ].filter(([, value]) => value);

  return (
    <div className="cv-clinical-result">
      <PendingModel
        label="Clinical risk model"
        note="There is no trained clinical prediction model, so these values are not scored and no risk level is computed."
        requirement="notebooks/03_Clinical_Model.ipynb needs to be written and run, with the fitted model saved to models/clinical/."
      />

      <div>
        <span className="cv-card-kicker">Recorded values</span>

        <p className="cv-muted-text">
          Passed to the language model as case context.
        </p>

        <div className="cv-clinical-values">
          {entries.map(([label, value]) => (
            <Metric
              key={label}
              label={label}
              value={value || "Not provided"}
              tone={value ? "" : "muted"}
            />
          ))}
        </div>

        {riskFactors.length > 0 && (
          <div className="cv-risk-chips">
            {riskFactors.map(([label]) => (
              <span key={label}>{label}</span>
            ))}
          </div>
        )}

        {clinicalData.symptoms && (
          <div className="cv-symptom-block">
            <span className="cv-card-kicker">Reported symptoms</span>

            <p>{clinicalData.symptoms}</p>
          </div>
        )}
      </div>
    </div>
  );
}

/* ============================================================
   METRIC
   ============================================================ */

function Metric({ label, value, tone = "" }) {
  return (
    <div className={`cv-metric ${tone}`}>
      <span>{label}</span>

      <strong>{value}</strong>
    </div>
  );
}

/* ============================================================
   FINDING
   ============================================================ */

function Finding({ title, value, muted }) {
  return (
    <div className={`cv-finding ${muted ? "muted" : ""}`}>
      <div className="cv-finding-check">{muted ? "○" : "✓"}</div>

      <div>
        <strong>{title}</strong>

        <span>{value}</span>
      </div>
    </div>
  );
}

/* ============================================================
   CONTEXT ITEM
   ============================================================ */

function ContextItem({ label, active, unavailable }) {
  return (
    <div
      className={`cv-context-item ${active ? "active" : ""} ${
        unavailable ? "unavailable" : ""
      }`}
    >
      <span>{active ? "✓" : unavailable ? "—" : "○"}</span>

      {label}

      {unavailable && <em>no model</em>}
    </div>
  );
}

/* ============================================================
   CHAT MESSAGE
   ============================================================ */

function ChatMessage({ message }) {
  const [showContext, setShowContext] = useState(false);

  return (
    <div
      className={`cv-message ${message.role} ${message.error ? "error" : ""}`}
    >
      <div className="cv-message-role">
        {message.role === "user" ? "You" : "CardioVision"}
      </div>

      <div className="cv-message-text">{message.text}</div>

      {message.role === "assistant" && !message.error && (
        <div className="cv-message-meta">
          <span>{message.model}</span>

          <span>·</span>

          <span>{message.device}</span>

          <span>·</span>

          <span>Local inference</span>

          {message.contextUsed && (
            <>
              <span>·</span>

              <button
                className="cv-context-toggle"
                onClick={() => setShowContext((previous) => !previous)}
              >
                {showContext ? "Hide case context" : "Show case context"}
              </button>
            </>
          )}
        </div>
      )}

      {showContext && message.contextPreview && (
        <pre className="cv-context-preview">{message.contextPreview}</pre>
      )}
    </div>
  );
}

export default App;
