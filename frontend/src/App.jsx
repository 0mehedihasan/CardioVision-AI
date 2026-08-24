import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import "./App.css";

import {
  analyzeEcho,
  askClinicalQuestion,
  deleteCase,
  fetchCase,
  fetchCaseImages,
  fetchCases,
  fetchHealth,
  fetchSession,
  login,
  logout,
  releaseImages,
  saveCase,
  setUnauthorizedHandler,
} from "./api";
import CaseList from "./components/CaseList";
import EchoResult from "./components/EchoResult";
import Login from "./components/Login";
import PatientForm from "./components/PatientForm";
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

const initialPatientData = {
  name: "",
  mrn: "",
  dateOfBirth: "",
  sex: "",
  studyDate: "",
  referringClinician: "",
  notes: "",
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

/* ============================================================
   APP — AUTHENTICATION GATE

   Nothing below this component renders until the backend has
   confirmed a session, and any 401 from any call drops straight
   back here.
   ============================================================ */

function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);

  const [session, setSession] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [authNotice, setAuthNotice] = useState(null);

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

  /* One handler for every expired or rejected token, registered on the API
     client so no individual call has to remember to check. */
  useEffect(() => {
    setUnauthorizedHandler((message) => {
      setSession(null);
      setAuthNotice(message);
    });

    return () => setUnauthorizedHandler(null);
  }, []);

  /* Revalidate a token left in sessionStorage by a page refresh. */
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const existing = await fetchSession();

        if (!cancelled && existing) {
          setSession(existing);
        }
      } catch {
        // A backend that is down or unreachable is not a valid session.
        // The login screen reports the real reason.
      } finally {
        if (!cancelled) setAuthChecked(true);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const signIn = useCallback(
    async (username, password) => {
      const data = await login(username, password);

      setAuthNotice(null);
      setSession(data);

      // The saved-case count and model state may have changed since the
      // login screen first loaded.
      loadHealth();
    },
    [loadHealth]
  );

  const signOut = useCallback(async () => {
    await logout();

    setSession(null);
    setAuthNotice(null);
  }, []);

  if (!authChecked) {
    return (
      <div className="cv-boot">
        <div className="cv-loader-spinner" />

        <span>Starting CardioVision…</span>
      </div>
    );
  }

  if (!session) {
    return (
      <>
        {authNotice && (
          <div className="cv-session-notice" role="status">
            {authNotice}
          </div>
        )}

        <Login
          health={health}
          healthError={healthError}
          onRetryHealth={loadHealth}
          onSignIn={signIn}
        />
      </>
    );
  }

  return (
    <Workspace
      session={session}
      health={health}
      healthError={healthError}
      onReloadHealth={loadHealth}
      onSignOut={signOut}
    />
  );
}

/* ============================================================
   WORKSPACE — SHELL, CASE LIST, CASE SWITCHING

   Owns the saved-case list and which case is open. The case
   itself lives in CaseWorkspace, keyed so that switching or
   starting a case remounts it. A remount is the only reset that
   cannot silently leave a stale field behind — which is exactly
   what made the old "New case" button do nothing.
   ============================================================ */

function Workspace({
  session,
  health,
  healthError,
  onReloadHealth,
  onSignOut,
}) {
  const [cases, setCases] = useState([]);
  const [casesTotal, setCasesTotal] = useState(0);
  const [casesLoading, setCasesLoading] = useState(true);
  const [casesError, setCasesError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");

  /* Bumping this key remounts CaseWorkspace with `openedCase` as its seed. */
  const [caseKey, setCaseKey] = useState(0);
  const [openedCase, setOpenedCase] = useState(null);

  const [activeSection, setActiveSection] = useState("case");
  const [switchError, setSwitchError] = useState(null);
  const [isSwitching, setIsSwitching] = useState(false);

  /* Published upward by the open case so the header can save it and warn
     about unsaved work without owning any of its state. */
  const [caseStatus, setCaseStatus] = useState(null);

  const storageReady = Boolean(health?.storage?.ready);

  const loadCases = useCallback(async (term = "") => {
    setCasesLoading(true);

    try {
      const data = await fetchCases(term);

      setCases(data.cases || []);
      setCasesTotal(data.total ?? (data.cases || []).length);
      setCasesError(null);
    } catch (error) {
      // A 401 is already handled globally; showing it here too would just
      // flash an error as the app returns to the login screen.
      if (error.status !== 401) {
        setCasesError(error.message);
      }
    } finally {
      setCasesLoading(false);
    }
  }, []);

  const handleSearch = useCallback(
    (term) => {
      setSearchTerm(term);
      loadCases(term);
    },
    [loadCases]
  );

  const refreshCases = useCallback(() => {
    loadCases(searchTerm);
  }, [loadCases, searchTerm]);

  const scrollToSection = useCallback((section) => {
    setActiveSection(section);

    document.getElementById(section)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }, []);

  /* ==========================================================
     UNSAVED-WORK GUARD

     Switching or resetting throws away whatever is in the form.
     Anything unsaved gets one confirmation first.
     ========================================================== */

  const confirmDiscard = useCallback(
    (action) => {
      if (!caseStatus?.dirty) return true;

      return window.confirm(
        `This case has unsaved changes. ${action} anyway?\n\n` +
          "Press Cancel to go back and save it first."
      );
    },
    [caseStatus]
  );

  const startNewCase = useCallback(() => {
    if (!confirmDiscard("Start a new case")) return;

    setOpenedCase(null);
    setSwitchError(null);
    setCaseStatus(null);
    setCaseKey((previous) => previous + 1);

    scrollToSection("case");
  }, [confirmDiscard, scrollToSection]);

  const openCase = useCallback(
    async (caseId) => {
      if (caseId === caseStatus?.caseId) {
        scrollToSection("case");
        return;
      }

      if (!confirmDiscard("Open another case")) return;

      setIsSwitching(true);
      setSwitchError(null);

      try {
        const record = await fetchCase(caseId);

        setOpenedCase(record);
        setCaseStatus(null);
        setCaseKey((previous) => previous + 1);

        scrollToSection("case");
      } catch (error) {
        if (error.status !== 401) {
          setSwitchError(error.message);
        }
      } finally {
        setIsSwitching(false);
      }
    },
    [caseStatus, confirmDiscard, scrollToSection]
  );

  const removeCase = useCallback(
    async (caseId) => {
      try {
        await deleteCase(caseId);

        // Deleting the case that is open would leave the form editing a
        // record that no longer exists, so clear it out.
        if (caseId === caseStatus?.caseId) {
          setOpenedCase(null);
          setCaseStatus(null);
          setCaseKey((previous) => previous + 1);
        }

        refreshCases();
        onReloadHealth();
      } catch (error) {
        if (error.status !== 401) {
          setSwitchError(error.message);
        }
      }
    },
    [caseStatus, refreshCases, onReloadHealth]
  );

  const initials = (session?.username || "user").slice(0, 2).toUpperCase();

  const echoModelReady = Boolean(health?.modalities?.echo?.available);
  const medgemmaReady = Boolean(health?.models?.medgemma?.loaded);
  const backendOnline = Boolean(health);

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
            className={`cv-header-button primary ${
              caseStatus?.saving ? "loading" : ""
            }`}
            disabled={
              !caseStatus?.canSave || caseStatus?.saving || !storageReady
            }
            title={
              storageReady
                ? "Save this case to the local database"
                : "The case database is unavailable"
            }
            onClick={() => caseStatus?.save?.()}
          >
            {caseStatus?.saving ? (
              <>
                <span className="cv-button-spinner" />
                Saving
              </>
            ) : caseStatus?.dirty || !caseStatus?.caseId ? (
              "Save case"
            ) : (
              "Saved"
            )}
          </button>

          <button className="cv-header-button" onClick={startNewCase}>
            New case
          </button>

          <div className="cv-user">
            <div className="cv-user-avatar" title={session?.username}>
              {initials}
            </div>

            <button className="cv-signout-button" onClick={onSignOut}>
              Sign out
            </button>
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

          <div className="cv-sidebar-divider" />

          <CaseList
            cases={cases}
            total={casesTotal}
            activeCaseId={caseStatus?.caseId || null}
            isLoading={casesLoading || isSwitching}
            error={casesError}
            onSearch={handleSearch}
            onOpen={openCase}
            onDelete={removeCase}
            onRefresh={refreshCases}
          />

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

        <CaseWorkspace
          key={caseKey}
          initialCase={openedCase}
          health={health}
          healthError={healthError}
          storageReady={storageReady}
          switchError={switchError}
          onReloadHealth={onReloadHealth}
          onCasesChanged={refreshCases}
          onStatus={setCaseStatus}
          onNewCase={startNewCase}
          scrollToSection={scrollToSection}
        />
      </div>
    </div>
  );
}

/* ============================================================
   CASE WORKSPACE

   Everything belonging to one patient case. Remounted whenever
   the open case changes, so its initial state is built once from
   `initialCase` and never has to be reconciled afterwards.
   ============================================================ */

function CaseWorkspace({
  initialCase,
  health,
  healthError,
  storageReady,
  switchError,
  onReloadHealth,
  onCasesChanged,
  onStatus,
  onNewCase,
  scrollToSection,
}) {
  /* ---- seeded from the loaded record, if any ---------------- */

  const [caseId, setCaseId] = useState(initialCase?.case_id || null);
  const [createdAt] = useState(initialCase?.created_at || null);
  const [savedAt, setSavedAt] = useState(initialCase?.updated_at || null);

  const [patientData, setPatientData] = useState(() =>
    // Every field is coerced to a string: SQLite hands back NULL for columns
    // that were never filled in, and a null `value` turns a controlled input
    // back into an uncontrolled one, which React then refuses to update.
    Object.fromEntries(
      Object.keys(initialPatientData).map((field) => [
        field,
        initialCase?.patient?.[field] ?? "",
      ])
    )
  );

  const [clinicalData, setClinicalData] = useState(() => {
    const stored = initialCase?.clinical || {};

    // Keyed off the blank form rather than spreading the stored object, so a
    // stray key from an older record cannot end up counting as clinical data.
    return Object.fromEntries(
      Object.entries(initialClinicalData).map(([field, fallback]) => [
        field,
        stored[field] ?? fallback,
      ])
    );
  });

  const [echoResult, setEchoResult] = useState(initialCase?.echo || null);
  const [echoRotate, setEchoRotate] = useState(
    initialCase?.echo?.orientation?.rotation_applied || 0
  );
  const [echoFlip, setEchoFlip] = useState(
    Boolean(initialCase?.echo?.orientation?.flip_applied)
  );
  const [analysisComplete, setAnalysisComplete] = useState(
    Boolean(initialCase?.echo)
  );
  const [activeResult, setActiveResult] = useState(
    initialCase?.echo ? "echo" : "overview"
  );

  const [conversation, setConversation] = useState(
    initialCase?.conversation || []
  );

  /* A restored case has no File objects — the upload itself is archived
     server-side, but the browser cannot reconstitute a File from it. The
     archived filename is kept so the UI can still say what was analysed. */
  const [files, setFiles] = useState({ echo: null, ccta: null, ecg: null });
  const restoredFilename =
    initialCase?.echo?.input?.filename || initialCase?.source_file || "";

  const [dragOver, setDragOver] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);

  const [question, setQuestion] = useState("");
  const [isAsking, setIsAsking] = useState(false);

  const [dirty, setDirty] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);

  const echoInput = useRef(null);
  const cctaInput = useRef(null);
  const ecgInput = useRef(null);

  const inputRefs = { echo: echoInput, ccta: cctaInput, ecg: ecgInput };

  const echoModelReady = Boolean(health?.modalities?.echo?.available);
  const medgemmaReady = Boolean(health?.models?.medgemma?.loaded);
  const backendOnline = Boolean(health);

  /* ==========================================================
     RESTORED IMAGES

     Stored renders come back as authenticated endpoints, which an
     <img> tag cannot fetch on its own. They are pulled as blobs
     and spliced into the restored result, then revoked on unmount
     so switching between cases does not leak them.
     ========================================================== */

  const blobUrls = useRef(null);

  useEffect(() => {
    const stored = initialCase?.images;

    if (!stored || Object.keys(stored).length === 0) return undefined;

    let cancelled = false;

    (async () => {
      try {
        const urls = await fetchCaseImages(
          initialCase.case_id,
          Object.keys(stored)
        );

        if (cancelled) {
          releaseImages(urls);
          return;
        }

        blobUrls.current = urls;

        setEchoResult((previous) =>
          previous ? { ...previous, images: urls } : previous
        );
      } catch {
        // The findings and measurements are already on screen; a failed
        // image fetch degrades the view rather than breaking the case.
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [initialCase]);

  useEffect(
    () => () => {
      releaseImages(blobUrls.current);
      blobUrls.current = null;
    },
    []
  );

  /* ==========================================================
     DERIVED
     ========================================================== */

  const hasClinicalData = Object.values(clinicalData).some(
    (value) => value !== "" && value !== false && value !== null
  );

  const hasPatientData = Object.values(patientData).some(
    (value) => typeof value === "string" && value.trim() !== ""
  );

  const canAnalyze = Boolean(files.echo) && echoModelReady && !isAnalyzing;

  /* Nothing to write is not an error, but it should not be a save either. */
  const canSave =
    hasPatientData ||
    hasClinicalData ||
    Boolean(echoResult) ||
    conversation.length > 0;

  const updatePatient = (field, value) => {
    setPatientData((previous) => ({ ...previous, [field]: value }));
    setDirty(true);
  };

  const updateClinical = (field, value) => {
    setClinicalData((previous) => ({ ...previous, [field]: value }));
    setDirty(true);
  };

  const handleFile = (modality, file) => {
    if (!file) return;

    setFiles((previous) => ({ ...previous, [modality]: file }));
    setDirty(true);

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
    setFiles((previous) => ({ ...previous, [modality]: null }));
    setDirty(true);

    if (modality === "echo") {
      setAnalysisComplete(false);
      setEchoResult(null);
      setAnalysisError(null);
      setEchoRotate(0);
      setEchoFlip(false);
    }
  };

  /* ==========================================================
     PERSISTENCE

     One writer for every save path — the explicit button, the
     automatic save around an analysis, and the save after an
     assistant reply — so a case can never be written two
     different ways.
     ========================================================== */

  const persist = useCallback(
    async (overrides = {}) => {
      const { silent = false, echo = echoResult, messages = conversation } =
        overrides;

      const payload = {
        case_id: overrides.caseId ?? caseId ?? undefined,
        patient: patientData,
        clinical: clinicalData,
        conversation: messages,
      };

      if (echo) {
        // The rendered PNGs travel separately: the backend decodes them to
        // files, so keeping the data URLs in echo_json would store every
        // image twice.
        const { images, ...rest } = echo;

        payload.echo = { ...rest, analyzed: true };

        if (images) {
          // Blob URLs come from a case that is already stored; only fresh
          // data URLs need writing.
          const encoded = Object.fromEntries(
            Object.entries(images).filter(
              ([, value]) =>
                typeof value === "string" && value.startsWith("data:")
            )
          );

          if (Object.keys(encoded).length > 0) {
            payload.images = encoded;
          }
        }
      }

      setIsSaving(true);

      if (!silent) setSaveError(null);

      try {
        const stored = await saveCase(payload);

        // Adopt the backend's ID and timestamps rather than guessing them.
        setCaseId(stored.case_id);
        setSavedAt(stored.updated_at);
        setDirty(false);
        setSaveError(null);

        onCasesChanged();

        return stored;
      } catch (error) {
        if (error.status !== 401) {
          setSaveError(error.message);
        }
        return null;
      } finally {
        setIsSaving(false);
      }
    },
    [caseId, patientData, clinicalData, echoResult, conversation, onCasesChanged]
  );

  /* Publish just enough for the header to drive Save and to warn before
     discarding. Everything else stays private to this component. */
  useEffect(() => {
    onStatus({
      caseId,
      dirty,
      saving: isSaving,
      canSave,
      save: () => persist(),
    });
  }, [caseId, dirty, isSaving, canSave, persist, onStatus]);

  /* ==========================================================
     ANALYSIS

     Calls POST /api/analyze/echo and renders the model's actual
     output. There are no simulated results anywhere in this flow.
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

      /* The case row has to exist before analysis so the backend can file
         the source image under it. Running a study is a real commitment,
         so creating the record at that moment is the honest behaviour —
         and it is what stops a completed analysis from evaporating on
         refresh. */
      let targetCase = caseId;

      if (!targetCase && storageReady) {
        const stored = await persist({ silent: true });
        targetCase = stored?.case_id || null;
      }

      try {
        const result = await analyzeEcho(files.echo, {
          rotate,
          flip,
          caseId: targetCase || undefined,
        });

        setEchoResult(result);
        setAnalysisComplete(true);
        setActiveResult("echo");

        if (storageReady) {
          await persist({
            silent: true,
            echo: result,
            caseId: targetCase || undefined,
          });
        } else {
          setDirty(true);
        }
      } catch (error) {
        setAnalysisError(error.message);
        // Refresh health so the UI reflects a backend that went away.
        onReloadHealth();
      } finally {
        setIsAnalyzing(false);
      }
    },
    [
      files.echo,
      echoModelReady,
      isAnalyzing,
      echoRotate,
      echoFlip,
      caseId,
      storageReady,
      persist,
      scrollToSection,
      onReloadHealth,
    ]
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
      case_id: caseId || "",
      patient: patientData,
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
        echo: Boolean(files.echo) || Boolean(echoResult),
        ccta: Boolean(files.ccta),
        ecg: Boolean(files.ecg),
      },
    }),
    [caseId, patientData, clinicalData, echoResult, files]
  );

  const askQuestion = async (text) => {
    const trimmed = (text ?? question).trim();

    if (!trimmed || isAsking) return;

    const withQuestion = [...conversation, { role: "user", text: trimmed }];

    setConversation(withQuestion);
    setQuestion("");
    setIsAsking(true);
    setDirty(true);

    try {
      const data = await askClinicalQuestion(trimmed, caseState);

      const answer = {
        role: "assistant",
        text:
          data.answer ||
          "No response was returned by the clinical language model.",
        model: data.model,
        device: data.device,
        contextUsed: data.context_used,
        contextPreview: data.context_preview,
      };

      const complete = [...withQuestion, answer];
      setConversation(complete);

      // Only fold the exchange into an existing record. Asking a general
      // cardiology question with no patient entered should not silently
      // create a case.
      if (caseId && storageReady) {
        persist({ silent: true, messages: complete });
      }
    } catch (error) {
      setConversation([
        ...withQuestion,
        { role: "assistant", text: error.message, error: true },
      ]);
    } finally {
      setIsAsking(false);
    }
  };

  const contextItems = [
    { label: "Patient", active: hasPatientData },
    { label: "Clinical", active: hasClinicalData },
    { label: "Echo findings", active: Boolean(echoResult) },
    { label: "CCTA", active: false, unavailable: true },
    { label: "ECG", active: false, unavailable: true },
  ];

  const savedLabel = savedAt ? formatTimestamp(savedAt) : "";

  return (
    <main className="cv-main">
      {/* ==================================================
          HERO
      ================================================== */}

      <section className="cv-hero">
        <div className="cv-eyebrow">Cardiovascular AI platform</div>

        <h1>Understand the heart through multimodal AI.</h1>

        <p>
          Echocardiography segmentation runs locally on a trained UNet++
          model. CCTA, ECG and clinical risk models are still in development
          and are clearly marked as unavailable.
        </p>
      </section>

      {!backendOnline && (
        <div className="cv-banner error">
          <strong>Backend offline</strong>

          <span>{healthError}</span>

          <button className="cv-secondary-button" onClick={onReloadHealth}>
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

          <button className="cv-secondary-button" onClick={onReloadHealth}>
            Recheck
          </button>
        </div>
      )}

      {backendOnline && !storageReady && (
        <div className="cv-banner warning">
          <strong>Case storage unavailable</strong>

          <span>
            {health?.storage?.error ||
              "The local case database could not be opened, so nothing can " +
                "be saved. Analysis still works, but this case will be lost " +
                "when the page reloads."}
          </span>
        </div>
      )}

      {switchError && (
        <div className="cv-banner error">
          <strong>Could not open that case</strong>

          <span>{switchError}</span>
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
              Provide the available patient information and imaging studies.
              Missing modalities can remain unavailable.
            </p>
          </div>

          <div className={`cv-case-chip ${caseId ? "saved" : ""}`}>
            <span>{caseId ? "Case ID" : "Status"}</span>

            <strong>{caseId || "New case"}</strong>

            {caseId && createdAt && (
              <em>Opened {formatTimestamp(createdAt)}</em>
            )}
          </div>
        </div>

        {saveError && (
          <div className="cv-banner error">
            <strong>Could not save this case</strong>

            <span>{saveError}</span>

            <button
              className="cv-secondary-button"
              disabled={isSaving}
              onClick={() => persist()}
            >
              Try again
            </button>
          </div>
        )}

        <div className="cv-case-grid three">
          <PatientForm
            data={patientData}
            onChange={updatePatient}
            caseId={caseId}
            isSaved={Boolean(savedAt) && !dirty}
            savedAt={savedLabel}
          />

          <ClinicalForm data={clinicalData} onChange={updateClinical} />

          <div className="cv-modality-panel">
            <div className="cv-panel-header">
              <span className="cv-card-kicker">Imaging studies</span>

              <h3>Modalities</h3>
            </div>

            {restoredFilename && !files.echo && (
              <div className="cv-restored-note">
                Restored from the saved record: <code>{restoredFilename}</code>.
                The findings below are the stored ones. Re-select the file to
                run the model again.
              </div>
            )}

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

          <div className="cv-analysis-actions">
            <button
              className="cv-secondary-button"
              disabled={!canSave || isSaving || !storageReady}
              onClick={() => persist()}
            >
              {isSaving ? "Saving…" : caseId ? "Save changes" : "Save case"}
            </button>

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
              Model output for every modality that has a trained model behind
              it.
            </p>
          </div>

          <div
            className={`cv-status-chip ${analysisComplete ? "complete" : ""}`}
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
                patientData={patientData}
                hasClinicalData={hasClinicalData}
                restoredFilename={restoredFilename}
              />
            )}

            {activeResult === "echo" &&
              (echoResult ? (
                <EchoResult
                  result={echoResult}
                  rotate={echoRotate}
                  flip={echoFlip}
                  onReorient={
                    files.echo ? reanalyzeWithOrientation : undefined
                  }
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
                left-ventricular-cavity probability was most sensitive to. It
                is a property of the model, not independent clinical
                evidence, and a bright region is not a finding.
              </div>
            </>
          ) : (
            /* An all-zero gradient renders as a smooth, plausible-looking
               heatmap. Showing it would be worse than showing nothing:
               it is an explanation of a computation that never happened. */
            <div className="cv-xai-unavailable">
              <strong>Attribution unavailable for this run</strong>

              <p>
                The gradient with respect to the input could not be computed,
                so no saliency map exists. The segmentation above is
                unaffected and remains valid — only the explanation is
                missing.
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
              Query the local medical language model. When a case has clinical
              data or echo findings, they are sent as context and the model is
              told which modalities are unavailable.
            </p>
          </div>

          <div className={`cv-local-badge ${medgemmaReady ? "" : "offline"}`}>
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
                {hasClinicalData || hasPatientData || echoResult
                  ? "Loaded"
                  : "Empty"}
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
              Unavailable modalities are named explicitly in the prompt so the
              model cannot invent findings for them. The patient's name and
              MRN are never sent.
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
                  Ask about general cardiology, or about the findings in this
                  case.
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

          <span>Multimodal cardiovascular imaging research platform</span>
        </div>

        <div>
          <button className="cv-link-button" onClick={onNewCase}>
            Start a new case
          </button>

          <span>Research prototype · Local inference</span>
        </div>
      </footer>
    </main>
  );
}

/* ============================================================
   TIMESTAMPS
   ============================================================ */

function formatTimestamp(iso) {
  if (!iso) return "";

  const when = new Date(iso);

  if (Number.isNaN(when.getTime())) return "";

  return when.toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/* ============================================================
   CLINICAL FORM
   ============================================================ */

function ClinicalForm({ data, onChange }) {
  return (
    <div className="cv-clinical-card">
      <div className="cv-panel-header">
        <span className="cv-card-kicker">Clinical data</span>

        <h3>Presentation</h3>
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
          Preprocessing the frame and running the UNet++ model, then computing
          gradient saliency. First run also loads the weights.
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

function OverviewResult({
  echoResult,
  files,
  clinicalData,
  patientData,
  hasClinicalData,
  restoredFilename,
}) {
  const structures = echoResult?.structures || [];
  const present = structures.filter((structure) => structure.present);
  const calibrated = Boolean(echoResult?.input?.has_spatial_calibration);

  // Derived from the model card rather than hardcoded, so adding a class
  // to the model cannot leave a stale denominator on screen.
  const foregroundCount = echoResult?.model?.num_classes
    ? echoResult.model.num_classes - 1
    : structures.length;

  const echoName =
    files.echo?.name || echoResult?.input?.filename || restoredFilename;

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
              Patient:{" "}
              {patientData?.name?.trim()
                ? patientData.name.trim()
                : "no name recorded"}
              {patientData?.mrn?.trim() ? ` · ${patientData.mrn.trim()}` : ""}
            </li>

            <li>
              Clinical data:{" "}
              {hasClinicalData
                ? `entered${clinicalData.age ? `, age ${clinicalData.age}` : ""}`
                : "none entered"}
            </li>

            <li>Echo image: {echoName || "none"}</li>

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
