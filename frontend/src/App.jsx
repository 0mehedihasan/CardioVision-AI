import { useMemo, useRef, useState } from "react";
import "./App.css";

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
  },
  ccta: {
    label: "Coronary CT angiography",
    short: "CCTA",
    description: "Cardiac CT / coronary anatomy",
    formats: "DICOM, NIfTI, ZIP series",
    accept: ".dcm,.nii,.nii.gz,.zip,.png,.jpg,.jpeg",
  },
  ecg: {
    label: "Electrocardiography",
    short: "ECG",
    description: "Waveform trace or report",
    formats: "PNG, JPEG, PDF, CSV",
    accept: ".png,.jpg,.jpeg,.pdf,.csv",
  },
};

const navItems = [
  { id: "case", number: "01", label: "Patient case" },
  { id: "results", number: "02", label: "Analysis" },
  { id: "explainability", number: "03", label: "Explainability" },
  { id: "assistant", number: "04", label: "Case assistant" },
];

const API_BASE_URL = "http://127.0.0.1:8000";

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

  const uploadedCount = Object.values(files).filter(Boolean).length;

  const hasClinicalData = Object.values(clinicalData).some(
    (value) => value !== "" && value !== false
  );

  const canAnalyze = uploadedCount > 0 || hasClinicalData;

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

    setAnalysisComplete(false);
  };

  const removeFile = (modality) => {
    setFiles((previous) => ({
      ...previous,
      [modality]: null,
    }));

    setAnalysisComplete(false);
  };

  const runAnalysis = () => {
    if (!canAnalyze || isAnalyzing) return;

    setIsAnalyzing(true);
    scrollToSection("results");

    setTimeout(() => {
      setIsAnalyzing(false);
      setAnalysisComplete(true);
      setActiveResult("overview");
    }, 2000);
  };

  /*
   * ============================================================
   * MEDGEMMA CLINICAL QUESTION
   * ============================================================
   */

  const askQuestion = async (text) => {
    const trimmed = (text ?? question).trim();

    if (!trimmed || isAsking) return;

    const userMessage = {
      role: "user",
      text: trimmed,
    };

    setConversation((previous) => [
      ...previous,
      userMessage,
    ]);

    setQuestion("");
    setIsAsking(true);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/clinical-question`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: trimmed,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Clinical question request failed with status ${response.status}`
        );
      }

      const data = await response.json();

      setConversation((previous) => [
        ...previous,
        {
          role: "assistant",
          text:
            data.answer ||
            "No response was returned by the clinical language model.",
          model: data.model || "MedGemma 1.5 4B IT",
          device: data.device || "mps",
        },
      ]);
    } catch (error) {
      console.error("Clinical question error:", error);

      setConversation((previous) => [
        ...previous,
        {
          role: "assistant",
          text:
            "Unable to connect to the local clinical model. Please make sure the CardioVision AI backend is running.",
          error: true,
        },
      ]);
    } finally {
      setIsAsking(false);
    }
  };

  const scrollToSection = (section) => {
    setActiveSection(section);

    const element = document.getElementById(section);

    if (element) {
      element.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  };

  return (
    <div className="cv-shell">
      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="cv-header">
        <div className="cv-brand">
          <div className="cv-brand-mark">CV</div>

          <div>
            <div className="cv-brand-name">
              CardioVision
            </div>

            <div className="cv-brand-subtitle">
              Multimodal cardiovascular intelligence
            </div>
          </div>
        </div>

        <div className="cv-header-center">
          <span className="cv-system-dot" />
          Local inference · system ready
        </div>

        <div className="cv-header-actions">
          <button
            className="cv-header-button"
            onClick={() => scrollToSection("case")}
          >
            New case
          </button>

          <div className="cv-user-avatar">
            MH
          </div>
        </div>
      </header>

      {/* ======================================================
          MOBILE NAV
      ====================================================== */}

      <nav className="cv-mobile-nav">
        {navItems.map((item) => (
          <button
            key={item.id}
            className={
              activeSection === item.id ? "active" : ""
            }
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
          <div className="cv-sidebar-label">
            Workspace
          </div>

          {navItems.map((item) => (
            <button
              key={item.id}
              className={`cv-nav-item ${
                activeSection === item.id
                  ? "active"
                  : ""
              }`}
              onClick={() => scrollToSection(item.id)}
            >
              <span className="cv-nav-number">
                {item.number}
              </span>

              <span>{item.label}</span>
            </button>
          ))}

          <div className="cv-sidebar-spacer" />

          <div className="cv-local-card">
            <div className="cv-local-icon">
              ◉
            </div>

            <div>
              <div className="cv-local-title">
                Local inference
              </div>

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
            <div className="cv-eyebrow">
              Cardiovascular AI platform
            </div>

            <h1>
              Understand the heart through multimodal AI.
            </h1>

            <p>
              Analyze clinical information, echocardiography,
              CCTA, and ECG data within a unified patient case.
            </p>
          </section>

          {/* ==================================================
              01. PATIENT CASE
          ================================================== */}

          <section id="case" className="cv-section">
            <div className="cv-section-head">
              <div>
                <div className="cv-section-index">
                  01 / Patient case
                </div>

                <h2>
                  Build a clinical case
                </h2>

                <p>
                  Provide the available patient information
                  and imaging studies. Missing modalities can
                  remain unavailable.
                </p>
              </div>

              <div className="cv-case-chip">
                <span>Case ID</span>
                <strong>{patientId}</strong>
              </div>
            </div>

            <div className="cv-case-grid">
              <ClinicalForm
                data={clinicalData}
                onChange={updateClinical}
              />

              <div className="cv-modality-panel">
                <div className="cv-panel-header">
                  <span className="cv-card-kicker">
                    Imaging studies
                  </span>

                  <h3>
                    Modalities
                  </h3>
                </div>

                <div className="cv-modality-list">
                  {Object.entries(modalityConfig).map(
                    ([key, modality]) => (
                      <ModalityRow
                        key={key}
                        modalityKey={key}
                        modality={modality}
                        file={files[key]}
                        inputRef={inputRefs[key]}
                        isDragOver={
                          dragOver === key
                        }
                        onDragOver={(over) =>
                          setDragOver(
                            over ? key : null
                          )
                        }
                        onUpload={(file) =>
                          handleFile(key, file)
                        }
                        onRemove={() =>
                          removeFile(key)
                        }
                      />
                    )
                  )}
                </div>
              </div>
            </div>

            <div className="cv-analysis-bar">
              <div>
                <div className="cv-analysis-title">
                  Ready for multimodal analysis
                </div>

                <div className="cv-analysis-subtitle">
                  {uploadedCount} imaging modalit
                  {uploadedCount === 1
                    ? "y"
                    : "ies"}{" "}
                  provided
                  {hasClinicalData
                    ? " · Clinical data available"
                    : " · No clinical data yet"}
                </div>
              </div>

              <button
                className={`cv-primary-button ${
                  isAnalyzing ? "loading" : ""
                }`}
                disabled={
                  !canAnalyze || isAnalyzing
                }
                onClick={runAnalysis}
              >
                {isAnalyzing ? (
                  <>
                    <span className="cv-button-spinner" />
                    Analyzing case
                  </>
                ) : (
                  <>
                    Analyze case
                    <span aria-hidden="true">
                      →
                    </span>
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
                <div className="cv-section-index">
                  02 / Multimodal analysis
                </div>

                <h2>
                  Case intelligence
                </h2>

                <p>
                  A unified workspace for modality-specific
                  inference and multimodal interpretation.
                </p>
              </div>

              <div
                className={`cv-status-chip ${
                  analysisComplete
                    ? "complete"
                    : ""
                }`}
              >
                <span />

                {analysisComplete
                  ? "Analysis complete"
                  : "Awaiting analysis"}
              </div>
            </div>

            {!analysisComplete &&
              !isAnalyzing && (
                <div className="cv-empty-analysis">
                  <div className="cv-empty-icon">
                    ◌
                  </div>

                  <h3>
                    No analysis generated yet
                  </h3>

                  <p>
                    Add patient information or imaging
                    data above, then run the analysis
                    pipeline.
                  </p>

                  <button
                    className="cv-secondary-button"
                    onClick={() =>
                      scrollToSection("case")
                    }
                  >
                    Configure patient case
                  </button>
                </div>
              )}

            {isAnalyzing && <AnalysisLoader />}

            {analysisComplete && (
              <div className="cv-results">
                <div
                  className="cv-tabs"
                  role="tablist"
                >
                  {[
                    ["overview", "Overview"],
                    ["echo", "Echo"],
                    ["ccta", "CCTA"],
                    ["clinical", "Clinical"],
                  ].map(([id, label]) => (
                    <button
                      key={id}
                      role="tab"
                      aria-selected={
                        activeResult === id
                      }
                      className={
                        activeResult === id
                          ? "active"
                          : ""
                      }
                      onClick={() =>
                        setActiveResult(id)
                      }
                    >
                      {label}
                    </button>
                  ))}
                </div>

                {activeResult === "overview" && (
                  <OverviewResult
                    files={files}
                    clinicalData={clinicalData}
                  />
                )}

                {activeResult === "echo" && (
                  <ModalityResult
                    file={files.echo}
                    label="Echocardiography"
                    metricLabel="LV segmentation"
                    metricValue="Available"
                    confidence="84.2%"
                    detail="The trained echocardiography model will provide cardiac structure segmentation and relevant prediction outputs here."
                  />
                )}

                {activeResult === "ccta" && (
                  <ModalityResult
                    file={files.ccta}
                    label="CCTA"
                    metricLabel="Anatomical analysis"
                    metricValue="Available"
                    confidence="89.1%"
                    detail="The CCTA model will provide coronary and cardiovascular structure segmentation together with disease-related imaging findings."
                  />
                )}

                {activeResult === "clinical" && (
                  <ClinicalResult
                    clinicalData={clinicalData}
                  />
                )}
              </div>
            )}
          </section>

          {/* ==================================================
              03. EXPLAINABILITY
          ================================================== */}

          {analysisComplete && (
            <section
              id="explainability"
              className="cv-section"
            >
              <div className="cv-section-head">
                <div>
                  <div className="cv-section-index">
                    03 / Explainability
                  </div>

                  <h2>
                    See why the model predicts
                  </h2>

                  <p>
                    Model explanations connect predictions
                    with relevant image regions and
                    anatomical structures.
                  </p>
                </div>
              </div>

              <div className="cv-xai-grid">
                <div className="cv-xai-image">
                  <div className="cv-image-placeholder cv-heatmap">
                    <div className="cv-heatmap-ring" />

                    <span className="cv-image-tag">
                      Grad-CAM
                    </span>
                  </div>

                  <div className="cv-image-caption">
                    <strong>
                      Prediction attribution
                    </strong>

                    <span>
                      Regions contributing to the selected
                      prediction
                    </span>
                  </div>
                </div>

                <div className="cv-xai-content">
                  <span className="cv-card-kicker">
                    Model explanation
                  </span>

                  <h3>
                    Coronary abnormality
                  </h3>

                  <div className="cv-confidence">
                    <div>
                      <span>
                        Model confidence
                      </span>

                      <strong>
                        87.4%
                      </strong>
                    </div>

                    <div className="cv-confidence-bar">
                      <span
                        style={{
                          width: "87.4%",
                        }}
                      />
                    </div>
                  </div>

                  <div className="cv-finding-list">
                    <Finding
                      title="Coronary anatomy"
                      value="Relevant region detected"
                    />

                    <Finding
                      title="Cardiac structure"
                      value="Left ventricular region"
                    />

                    <Finding
                      title="Model attention"
                      value="High regional contribution"
                    />
                  </div>

                  <div className="cv-xai-note">
                    Explainability outputs show model
                    attribution and should not be interpreted
                    as independent clinical evidence.
                  </div>
                </div>
              </div>

              <div className="cv-segmentation-card">
                <div>
                  <span className="cv-card-kicker">
                    Structural analysis
                  </span>

                  <h3>
                    Cardiovascular segmentation
                  </h3>

                  <p>
                    Predicted anatomical structures can be
                    visualized alongside the original
                    imaging data.
                  </p>
                </div>

                <div className="cv-segmentation-visual">
                  <div className="cv-anatomy-placeholder">
                    <span />
                    <span />
                    <span />
                  </div>

                  <div className="cv-segmentation-legend">
                    <span>
                      <i /> Coronary vessel
                    </span>

                    <span>
                      <i /> Cardiac structure
                    </span>

                    <span>
                      <i /> Segmentation region
                    </span>
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* ==================================================
              04. CASE ASSISTANT
          ================================================== */}

          <section
            id="assistant"
            className="cv-section cv-assistant-section"
          >
            <div className="cv-section-head">
              <div>
                <div className="cv-section-index">
                  04 / Case assistant
                </div>

                <h2>
                  Ask about this case
                </h2>

                <p>
                  Query the local medical language model
                  using the available case context and
                  model-generated findings.
                </p>
              </div>

              <div className="cv-local-badge">
                <span />
                Local medical model
              </div>
            </div>

            <div className="cv-assistant">
              {/* ============================================
                  CASE CONTEXT
              ============================================ */}

              <div className="cv-assistant-context">
                <div className="cv-assistant-context-header">
                  <span className="cv-card-kicker">
                    Case context
                  </span>

                  <span className="cv-context-status">
                    Loaded
                  </span>
                </div>

                <div className="cv-context-items">
                  <ContextItem
                    label="Clinical"
                    active={hasClinicalData}
                  />

                  <ContextItem
                    label="Echocardiography"
                    active={Boolean(files.echo)}
                  />

                  <ContextItem
                    label="CCTA"
                    active={Boolean(files.ccta)}
                  />

                  <ContextItem
                    label="ECG"
                    active={Boolean(files.ecg)}
                  />

                  <ContextItem
                    label="AI findings"
                    active={analysisComplete}
                  />
                </div>
              </div>

              {/* ============================================
                  CHAT
              ============================================ */}

              <div className="cv-chat">
                {conversation.length === 0 ? (
                  <div className="cv-chat-empty">
                    <div className="cv-chat-icon">
                      +
                    </div>

                    <h3>
                      Ask a clinical question
                    </h3>

                    <p>
                      Ask the local medical language model
                      to explain or answer a clinical
                      question.
                    </p>

                    <div className="cv-question-suggestions">
                      <button
                        onClick={() =>
                          askQuestion(
                            "What is hypertension?"
                          )
                        }
                        disabled={isAsking}
                      >
                        What is hypertension?
                      </button>

                      <button
                        onClick={() =>
                          askQuestion(
                            "What are the major risk factors for coronary artery disease?"
                          )
                        }
                        disabled={isAsking}
                      >
                        CAD risk factors
                      </button>

                      <button
                        onClick={() =>
                          askQuestion(
                            "What findings require further clinical attention?"
                          )
                        }
                        disabled={isAsking}
                      >
                        What needs attention?
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="cv-conversation">
                    {conversation.map(
                      (message, index) => (
                        <ChatMessage
                          key={index}
                          message={message}
                        />
                      )
                    )}

                    {isAsking && (
                      <div className="cv-message assistant">
                        <div className="cv-message-role">
                          CardioVision
                        </div>

                        <div className="cv-message-text cv-thinking">
                          <span />
                          <span />
                          <span />
                          <em>
                            MedGemma is generating...
                          </em>
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
                      isAsking
                        ? "MedGemma is generating a response..."
                        : "Ask a clinical question..."
                    }
                    value={question}
                    disabled={isAsking}
                    onChange={(event) =>
                      setQuestion(
                        event.target.value
                      )
                    }
                    onKeyDown={(event) => {
                      if (
                        event.key === "Enter" &&
                        !event.shiftKey
                      ) {
                        event.preventDefault();

                        if (!isAsking) {
                          askQuestion();
                        }
                      }
                    }}
                  />

                  <button
                    onClick={() =>
                      askQuestion()
                    }
                    disabled={
                      !question.trim() ||
                      isAsking
                    }
                  >
                    {isAsking ? (
                      <>
                        <span className="cv-button-spinner" />
                        Asking
                      </>
                    ) : (
                      <>
                        Ask
                        <span aria-hidden="true">
                          →
                        </span>
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
              <strong>
                CardioVision AI
              </strong>

              <span>
                Multimodal cardiovascular imaging
                research platform
              </span>
            </div>

            <div>
              Research prototype · Local inference
            </div>
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
        <span className="cv-card-kicker">
          Clinical data
        </span>

        <h3>
          Patient information
        </h3>
      </div>

      <div className="cv-form">
        <div className="cv-form-row">
          <label>
            Age

            <input
              type="number"
              placeholder="e.g. 58"
              value={data.age}
              onChange={(event) =>
                onChange(
                  "age",
                  event.target.value
                )
              }
            />
          </label>

          <label>
            Sex

            <select
              value={data.sex}
              onChange={(event) =>
                onChange(
                  "sex",
                  event.target.value
                )
              }
            >
              <option value="">
                Select
              </option>

              <option value="Male">
                Male
              </option>

              <option value="Female">
                Female
              </option>
            </select>
          </label>
        </div>

        <label>
          Primary symptoms

          <textarea
            rows="3"
            placeholder="Chest pain, dyspnea, fatigue..."
            value={data.symptoms}
            onChange={(event) =>
              onChange(
                "symptoms",
                event.target.value
              )
            }
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
                onChange(
                  "bloodPressure",
                  event.target.value
                )
              }
            />
          </label>

          <label>
            Heart rate

            <input
              type="number"
              placeholder="72 bpm"
              value={data.heartRate}
              onChange={(event) =>
                onChange(
                  "heartRate",
                  event.target.value
                )
              }
            />
          </label>
        </div>

        <div className="cv-checkbox-grid">
          <label className="cv-checkbox">
            <input
              type="checkbox"
              checked={data.diabetes}
              onChange={(event) =>
                onChange(
                  "diabetes",
                  event.target.checked
                )
              }
            />

            <span>
              Diabetes
            </span>
          </label>

          <label className="cv-checkbox">
            <input
              type="checkbox"
              checked={data.hypertension}
              onChange={(event) =>
                onChange(
                  "hypertension",
                  event.target.checked
                )
              }
            />

            <span>
              Hypertension
            </span>
          </label>

          <label className="cv-checkbox">
            <input
              type="checkbox"
              checked={data.smoking}
              onChange={(event) =>
                onChange(
                  "smoking",
                  event.target.checked
                )
              }
            />

            <span>
              Smoking
            </span>
          </label>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   MODALITY ROW
   ============================================================ */

function ModalityRow({
  modalityKey,
  modality,
  file,
  inputRef,
  isDragOver,
  onDragOver,
  onUpload,
  onRemove,
}) {
  const status = file
    ? "ready"
    : "empty";

  return (
    <div
      className={`cv-modality-row ${status} ${
        isDragOver ? "dragging" : ""
      }`}
      onDragOver={(event) => {
        event.preventDefault();
        onDragOver(true);
      }}
      onDragLeave={() =>
        onDragOver(false)
      }
      onDrop={(event) => {
        event.preventDefault();

        onDragOver(false);

        onUpload(
          event.dataTransfer.files?.[0]
        );
      }}
    >
      <div className="cv-modality-icon">
        {modality.short}
      </div>

      <div className="cv-modality-info">
        <div className="cv-modality-title">
          {modality.label}
        </div>

        <div className="cv-modality-description">
          {file
            ? file.name
            : modality.description}
        </div>

        {!file && (
          <div className="cv-modality-formats">
            {modality.formats}
          </div>
        )}

        {file && (
          <div className="cv-modality-formats">
            {(file.size / 1024 / 1024).toFixed(
              2
            )}{" "}
            MB
          </div>
        )}
      </div>

      <div className="cv-modality-actions">
        {file ? (
          <>
            <span className="cv-ready-badge">
              ✓ Loaded
            </span>

            <button
              className="cv-ghost-button"
              onClick={() =>
                inputRef.current?.click()
              }
            >
              Replace
            </button>

            <button
              className="cv-ghost-button danger"
              onClick={onRemove}
            >
              Remove
            </button>
          </>
        ) : (
          <button
            className="cv-ghost-button"
            onClick={() =>
              inputRef.current?.click()
            }
          >
            Select file
          </button>
        )}

        <input
          ref={inputRef}
          type="file"
          accept={modality.accept}
          hidden
          onChange={(event) =>
            onUpload(
              event.target.files?.[0]
            )
          }
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
        <h3>
          Running multimodal analysis
        </h3>

        <p>
          Processing the available patient data
          through modality-specific models and
          preparing the fusion context.
        </p>
      </div>

      <div className="cv-loader-steps">
        <LoaderStep
          number="01"
          text="Preprocessing"
          active
        />

        <LoaderStep
          number="02"
          text="Modality inference"
          active
        />

        <LoaderStep
          number="03"
          text="Multimodal reasoning"
        />

        <LoaderStep
          number="04"
          text="Explainability"
        />
      </div>
    </div>
  );
}

function LoaderStep({
  number,
  text,
  active,
}) {
  return (
    <div
      className={`cv-loader-step ${
        active ? "active" : ""
      }`}
    >
      <span>
        {number}
      </span>

      {text}
    </div>
  );
}

/* ============================================================
   OVERVIEW RESULT
   ============================================================ */

function OverviewResult({
  files,
  clinicalData,
}) {
  return (
    <div className="cv-overview">
      <div className="cv-summary-card">
        <div className="cv-summary-main">
          <span className="cv-card-kicker">
            Primary model finding
          </span>

          <h3>
            Cardiovascular abnormality detected
          </h3>

          <p>
            The available case information indicates
            findings that warrant further clinical review.
            This result represents a research model output.
          </p>
        </div>

        <div className="cv-summary-confidence">
          <span>
            Confidence
          </span>

          <strong>
            87.4%
          </strong>
        </div>
      </div>

      <div className="cv-metric-grid">
        <Metric
          label="Clinical risk"
          value="Elevated"
          tone="warning"
        />

        <Metric
          label="Echo finding"
          value={
            files.echo
              ? "Analyzed"
              : "Awaiting input"
          }
          tone={
            files.echo
              ? ""
              : "muted"
          }
        />

        <Metric
          label="CCTA finding"
          value={
            files.ccta
              ? "Analyzed"
              : "Awaiting input"
          }
          tone={
            files.ccta
              ? ""
              : "muted"
          }
        />

        <Metric
          label="ECG"
          value={
            files.ecg
              ? "Analyzed"
              : "Awaiting input"
          }
          tone={
            files.ecg
              ? ""
              : "muted"
          }
        />
      </div>

      <div className="cv-findings-grid">
        <div className="cv-finding-card">
          <span className="cv-card-kicker">
            Key findings
          </span>

          <Finding
            title="Coronary anatomy"
            value="Potential abnormality"
          />

          <Finding
            title="Cardiac function"
            value="Requires review"
          />

          <Finding
            title="Clinical profile"
            value={
              clinicalData.hypertension
                ? "Hypertension reported"
                : "Available data reviewed"
            }
          />
        </div>

        <div className="cv-finding-card cv-next-step">
          <span className="cv-card-kicker">
            Next step
          </span>

          <h3>
            Review model evidence
          </h3>

          <p>
            Inspect modality-specific outputs and
            explainability visualizations before drawing
            any clinical conclusion.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   MODALITY RESULT
   ============================================================ */

function ModalityResult({
  file,
  label,
  metricLabel,
  metricValue,
  confidence,
  detail,
}) {
  return (
    <div className="cv-modality-result">
      <div className="cv-result-visual">
        <div className="cv-result-visual-image">
          <div className="cv-scan-grid">
            <div />
            <div />
            <div />
          </div>

          <span className="cv-image-tag">
            Model output
          </span>
        </div>

        <div className="cv-image-caption">
          <strong>
            {label} analysis
          </strong>

          <span>
            {file
              ? `Analyzed: ${file.name}`
              : `No ${label.toLowerCase()} file provided`}
          </span>
        </div>
      </div>

      <div className="cv-result-detail">
        <span className="cv-card-kicker">
          {label} model
        </span>

        <h3>
          Structural analysis
        </h3>

        <p>
          {detail}
        </p>

        <div className="cv-result-stat-row">
          <Metric
            label={metricLabel}
            value={metricValue}
          />

          <Metric
            label="Model confidence"
            value={confidence}
          />
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   CLINICAL RESULT
   ============================================================ */

function ClinicalResult({
  clinicalData,
}) {
  return (
    <div className="cv-clinical-result">
      <div>
        <span className="cv-card-kicker">
          Clinical model
        </span>

        <h3>
          Patient-level clinical assessment
        </h3>

        <p>
          Structured clinical variables are processed by
          the clinical prediction model and contribute to
          the multimodal case representation.
        </p>
      </div>

      <div className="cv-clinical-values">
        <Metric
          label="Age"
          value={
            clinicalData.age ||
            "Not provided"
          }
          tone={
            clinicalData.age
              ? ""
              : "muted"
          }
        />

        <Metric
          label="Sex"
          value={
            clinicalData.sex ||
            "Not provided"
          }
          tone={
            clinicalData.sex
              ? ""
              : "muted"
          }
        />

        <Metric
          label="Blood pressure"
          value={
            clinicalData.bloodPressure ||
            "Not provided"
          }
          tone={
            clinicalData.bloodPressure
              ? ""
              : "muted"
          }
        />

        <Metric
          label="Heart rate"
          value={
            clinicalData.heartRate
              ? `${clinicalData.heartRate} bpm`
              : "Not provided"
          }
          tone={
            clinicalData.heartRate
              ? ""
              : "muted"
          }
        />
      </div>
    </div>
  );
}

/* ============================================================
   METRIC
   ============================================================ */

function Metric({
  label,
  value,
  tone = "",
}) {
  return (
    <div
      className={`cv-metric ${tone}`}
    >
      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>
    </div>
  );
}

/* ============================================================
   FINDING
   ============================================================ */

function Finding({
  title,
  value,
}) {
  return (
    <div className="cv-finding">
      <div className="cv-finding-check">
        ✓
      </div>

      <div>
        <strong>
          {title}
        </strong>

        <span>
          {value}
        </span>
      </div>
    </div>
  );
}

/* ============================================================
   CONTEXT ITEM
   ============================================================ */

function ContextItem({
  label,
  active,
}) {
  return (
    <div
      className={`cv-context-item ${
        active ? "active" : ""
      }`}
    >
      <span>
        {active ? "✓" : "○"}
      </span>

      {label}
    </div>
  );
}

/* ============================================================
   CHAT MESSAGE
   ============================================================ */

function ChatMessage({
  message,
}) {
  return (
    <div
      className={`cv-message ${
        message.role
      } ${message.error ? "error" : ""}`}
    >
      <div className="cv-message-role">
        {message.role === "user"
          ? "You"
          : "CardioVision"}
      </div>

      <div className="cv-message-text">
        {message.text}
      </div>

      {message.role === "assistant" &&
        !message.error && (
          <div className="cv-message-meta">
            <span>
              {message.model ||
                "MedGemma 1.5 4B IT"}
            </span>

            <span>·</span>

            <span>
              {message.device || "mps"}
            </span>

            <span>·</span>

            <span>
              Local inference
            </span>
          </div>
        )}

      {message.evidence && (
        <div className="cv-message-evidence">
          <div className="cv-evidence-row">
            <span>
              Evidence
            </span>

            <strong>
              {message.evidence.finding}
            </strong>
          </div>

          <div className="cv-evidence-row">
            <span>
              Model confidence
            </span>

            <strong>
              {message.evidence.confidence}
            </strong>
          </div>

          {message.evidence.context
            ?.length > 0 && (
            <div className="cv-evidence-row">
              <span>
                Clinical context
              </span>

              <strong>
                {message.evidence.context.join(
                  " · "
                )}
              </strong>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;