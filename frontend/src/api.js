/*
 * ============================================================
 * CardioVision AI — backend client
 *
 * Every call except fetchHealth and login carries the session token.
 * A 401 anywhere triggers the onUnauthorized hook so the app can drop
 * straight back to the login screen instead of showing a wall of
 * "not signed in" errors on every panel.
 * ============================================================
 */

/*
 * The backend origin. Baked in at build time, because that is how Vite
 * exposes configuration — there is no runtime lookup to do.
 *
 * The default is the loopback address the backend binds by default, which is
 * the only address it should be reachable on: one shared password and no TLS.
 * Override with VITE_API_BASE_URL in frontend/.env.local when the API runs on
 * another port, and drop the trailing slash if you add one, since every path
 * below starts with one.
 */
export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

/* ============================================================
   TOKEN
   ============================================================ */

const TOKEN_KEY = "cardiovision.token";

let authToken = null;
let onUnauthorized = null;

/*
 * sessionStorage, not localStorage: the token dies with the browser tab.
 * A clinical workstation left open should not still be signed in tomorrow
 * because a token persisted to disk.
 */
function readStoredToken() {
  try {
    return window.sessionStorage.getItem(TOKEN_KEY);
  } catch {
    // Private browsing or a hardened profile can throw on access. The app
    // still works, it just will not survive a refresh.
    return null;
  }
}

export function getToken() {
  if (authToken === null) {
    authToken = readStoredToken();
  }

  return authToken;
}

export function setToken(token) {
  authToken = token || null;

  try {
    if (token) {
      window.sessionStorage.setItem(TOKEN_KEY, token);
    } else {
      window.sessionStorage.removeItem(TOKEN_KEY);
    }
  } catch {
    // Non-fatal: the in-memory copy is what every request actually uses.
  }
}

/** Register a callback fired whenever the backend rejects the token. */
export function setUnauthorizedHandler(handler) {
  onUnauthorized = handler;
}

/* ============================================================
   ERRORS
   ============================================================ */

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * FastAPI returns errors as { detail: string } or { detail: [...] }.
 * Pull out something a human can act on.
 */
async function readError(response, fallback) {
  let detail = "";

  try {
    const body = await response.json();

    if (typeof body?.detail === "string") {
      detail = body.detail;
    } else if (Array.isArray(body?.detail)) {
      detail = body.detail
        .map((item) => item?.msg || JSON.stringify(item))
        .join("; ");
    }
  } catch {
    // Body was not JSON — fall through to the generic message.
  }

  return detail || `${fallback} (HTTP ${response.status})`;
}

/*
 * The command in this message is the one the operator will actually type, so
 * it has to match the installed package rather than the layout the backend
 * once had. `cardiovision serve` is the console script declared in
 * pyproject.toml; the older `uvicorn main:app` form named a module that no
 * longer exists, which sent anyone following it to a directory that is not
 * there any more.
 */
function offlineMessage() {
  return (
    "Cannot reach the CardioVision backend at " +
    `${API_BASE_URL}. Start it with: cardiovision serve ` +
    "(after pip install -e . in the project root)."
  );
}

/* ============================================================
   CORE REQUEST
   ============================================================ */

/**
 * One place where the token is attached and a 401 is handled, so no
 * individual call can forget either.
 */
async function request(path, options = {}) {
  const { auth = true, fallback = "Request failed", ...rest } = options;

  const headers = { ...(rest.headers || {}) };

  if (auth) {
    const token = getToken();

    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
  }

  let response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, { ...rest, headers });
  } catch (error) {
    if (error instanceof TypeError) {
      throw new ApiError(offlineMessage(), 0);
    }
    throw error;
  }

  if (response.status === 401 && auth) {
    // The token is dead either way, so clear it before anything else can
    // retry with it.
    setToken(null);

    const message = await readError(response, "Session expired");

    if (onUnauthorized) {
      onUnauthorized(message);
    }

    throw new ApiError(message, 401);
  }

  if (!response.ok) {
    throw new ApiError(await readError(response, fallback), response.status);
  }

  return response;
}

/* ============================================================
   AUTHENTICATION
   ============================================================ */

export async function login(username, password) {
  const response = await request("/api/auth/login", {
    auth: false,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
    fallback: "Sign-in failed",
  });

  const data = await response.json();

  setToken(data.token);

  return data;
}

export async function logout() {
  try {
    await request("/api/auth/logout", {
      method: "POST",
      fallback: "Sign-out failed",
    });
  } catch {
    // A failed logout still means this browser forgets the token, which is
    // the part the operator cares about.
  } finally {
    setToken(null);
  }
}

/** Validate a stored token on page load. Returns null when there is none. */
export async function fetchSession() {
  if (!getToken()) return null;

  try {
    const response = await request("/api/auth/session", {
      fallback: "Could not verify the session",
    });

    return await response.json();
  } catch (error) {
    if (error.status === 401) return null;
    throw error;
  }
}

/* ============================================================
   HEALTH
   ============================================================ */

export async function fetchHealth() {
  // Public, so the login screen can report backend and model state before
  // anyone signs in.
  const response = await request("/api/health", {
    auth: false,
    fallback: "Health check failed",
  });

  return await response.json();
}

/* ============================================================
   CCTA SEGMENTATION
   ============================================================ */

/**
 * Upload one coronary CT volume and segment the lumen.
 *
 * This is the slow call in the application. A 0.5 mm study resampled to
 * 1 mm needs several hundred sliding windows, which is minutes of CPU, so
 * there is deliberately no client-side timeout — aborting at ten seconds
 * would make every real study look like a failure.
 *
 * `maxWindows` is a compute budget, not a quality setting. When the budget
 * cannot cover the volume the backend analyses a centred crop and reports
 * `coverage.complete: false` with the percentage it actually examined. That
 * is a first-class result and the caller must render it as "not looked at",
 * never as "nothing found".
 *
 * @param {File} file      .nii, .nii.gz, or a .zip holding one DICOM series
 * @param {object} options
 * @param {number} [options.maxWindows]        sliding-window budget
 * @param {boolean} [options.includeGradcam]   compute 3-D Grad-CAM
 * @param {boolean} [options.includeFigures]   render the slice panels
 * @param {string} [options.caseId]            archive the upload with this case
 */
export async function analyzeCcta(file, options = {}) {
  const {
    maxWindows,
    includeGradcam = true,
    includeFigures = true,
    caseId,
  } = options;

  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams();
  params.set("include_gradcam", String(includeGradcam));
  params.set("include_figures", String(includeFigures));

  // Omitted rather than guessed, so the server's own default is what applies
  // and the request URL shows whether the operator overrode it.
  if (Number.isFinite(maxWindows) && maxWindows > 0) {
    params.set("max_windows", String(Math.trunc(maxWindows)));
  }

  if (caseId) {
    params.set("case_id", caseId);
  }

  const response = await request(`/api/analyze/ccta?${params.toString()}`, {
    method: "POST",
    body: formData,
    fallback: "CCTA segmentation failed",
  });

  return await response.json();
}

/* ============================================================
   ECHO SEGMENTATION
   ============================================================ */

/**
 * Upload one echocardiography image and run segmentation.
 *
 * @param {File} file      PNG, JPEG, NIfTI or DICOM
 * @param {object} options
 * @param {number} [options.frame]        frame index for DICOM cine loops
 * @param {number} [options.rotate]       0 | 90 | 180 | 270, counter-clockwise
 * @param {boolean} [options.flip]        mirror horizontally
 * @param {boolean} [options.includeMask] request the raw mask array
 * @param {string} [options.caseId]       archive the upload with this case
 */
export async function analyzeEcho(file, options = {}) {
  const {
    frame,
    rotate = 0,
    flip = false,
    includeMask = true,
    caseId,
  } = options;

  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams();
  params.set("include_mask", String(includeMask));

  if (Number.isInteger(frame)) {
    params.set("frame", String(frame));
  }

  // Only sent when non-default, so the request URL shows at a glance
  // whether the image was transformed.
  if (rotate) {
    params.set("rotate", String(rotate));
  }

  if (flip) {
    params.set("flip", "true");
  }

  if (caseId) {
    params.set("case_id", caseId);
  }

  const response = await request(
    `/api/analyze/echo?${params.toString()}`,
    {
      method: "POST",
      body: formData,
      fallback: "Echo segmentation failed",
    }
  );

  return await response.json();
}

/* ============================================================
   ECG CLASSIFICATION
   ============================================================ */

/**
 * Upload one 12-lead ECG recording and classify it.
 *
 * WFDB — the format PTB-XL ships in, and so the format this model was
 * trained on — splits a recording across a `.hea` describing the layout
 * and a `.dat` holding the samples, and neither is readable alone. The
 * primary `file` is therefore allowed companions, which the backend
 * matches by filename rather than by position: browsers are free to
 * reorder multipart fields, and an off-by-one here would pair a header
 * with the wrong signal file and still decode.
 *
 * @param {File} file      .hea, .csv, .npy, .json or a .zip of the set
 * @param {object} options
 * @param {File[]} [options.companions]        other files of the same record
 * @param {number} [options.samplingFrequency] Hz, for formats that omit it
 * @param {string} [options.targetClass]       class the saliency explains
 * @param {boolean} [options.includeFigures]   render the strip and the chart
 * @param {string} [options.caseId]            archive the upload with this case
 */
export async function analyzeEcg(file, options = {}) {
  const {
    companions = [],
    samplingFrequency,
    targetClass,
    includeFigures = true,
    caseId,
  } = options;

  const formData = new FormData();
  formData.append("file", file);

  for (const companion of companions) {
    if (companion) {
      formData.append("companions", companion);
    }
  }

  const params = new URLSearchParams();
  params.set("include_figures", String(includeFigures));

  // Only sent when supplied. Guessing a rate would rescale the whole
  // recording in time — a 500 Hz record read as 100 Hz becomes a 50-second
  // strip resampled down to 10, which the model will still classify.
  if (Number.isFinite(samplingFrequency) && samplingFrequency > 0) {
    params.set("sampling_frequency", String(samplingFrequency));
  }

  if (targetClass) {
    params.set("target_class", targetClass);
  }

  if (caseId) {
    params.set("case_id", caseId);
  }

  const response = await request(`/api/analyze/ecg?${params.toString()}`, {
    method: "POST",
    body: formData,
    fallback: "ECG classification failed",
  });

  return await response.json();
}

/* ============================================================
   CLINICAL QUESTION
   ============================================================ */

/**
 * Ask MedGemma a question, optionally scoped to the current case.
 *
 * `caseState` is sent structured; the backend renders it into prompt
 * text so prompt construction stays server-side.
 */
export async function askClinicalQuestion(question, caseState = null) {
  const response = await request("/api/clinical-question", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, case: caseState }),
    fallback: "Clinical question failed",
  });

  return await response.json();
}

/* ============================================================
   INTEGRATED EVIDENCE AND REPORTS
   ============================================================ */

/*
 * Both calls below send the case the browser is holding rather than a
 * case_id. Analyses exist on screen before anyone presses save, and making
 * the report load from storage would show the operator a report for a
 * version of the case they can no longer see. The backend accepts either
 * form and prefers the body when both arrive.
 */

/**
 * Aggregate the current case into structured evidence.
 *
 * No language model is involved and no model is re-run — this reads the
 * analyses that already happened. It therefore answers correctly on a
 * backend where nothing loaded, which is exactly when the operator most
 * needs to see what is missing.
 *
 * @param {object} caseState  { case_id?, patient?, clinical?, ccta?, echo?, ecg? }
 */
export async function integratedEvidence(caseState) {
  const response = await request("/api/evidence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      case_id: caseState?.case_id || null,
      case: caseState || null,
    }),
    fallback: "Could not aggregate the case evidence",
  });

  return await response.json();
}

/**
 * Build the structured clinical report for the current case.
 *
 * The structured report is assembled first and the narrative is written into
 * it afterwards, so a MedGemma failure returns 200 with `ai_summary: null`
 * and `ai_summary_error` set. Render that as a missing summary beside intact
 * findings — not as a failed report.
 *
 * @param {object} caseState
 * @param {object} options
 * @param {boolean} [options.includeSummary]  have MedGemma write the narrative
 * @param {boolean} [options.includePrompt]   return the exact prompt it was given
 * @param {boolean} [options.save]            store the report against the case
 */
export async function generateReport(caseState, options = {}) {
  const {
    includeSummary = true,
    includePrompt = false,
    save = false,
  } = options;

  const params = new URLSearchParams();
  params.set("include_summary", String(includeSummary));
  params.set("include_prompt", String(includePrompt));
  params.set("save", String(save));

  const response = await request(`/api/report?${params.toString()}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      case_id: caseState?.case_id || null,
      case: caseState || null,
    }),
    fallback: "Could not build the report",
  });

  return await response.json();
}

/**
 * The last report saved for a case, or null when none was ever saved.
 *
 * "This case has not been reported on" is a normal state and the endpoint
 * returns `report: null` for it rather than a 404, so an absent report is
 * distinguishable from an absent case.
 */
export async function fetchStoredReport(caseId) {
  if (!caseId) return null;

  const response = await request(
    `/api/cases/${encodeURIComponent(caseId)}/report`,
    { fallback: "Could not load the saved report" }
  );

  const data = await response.json();

  return data.report || null;
}

/* ============================================================
   CASES
   ============================================================ */

export async function fetchCases(search = "") {
  const params = new URLSearchParams();

  if (search.trim()) {
    params.set("search", search.trim());
  }

  const query = params.toString();

  const response = await request(
    `/api/cases${query ? `?${query}` : ""}`,
    { fallback: "Could not load saved cases" }
  );

  return await response.json();
}

/**
 * Create or update a case.
 *
 * Omit case_id to mint a new one. The response carries the stored record,
 * so the caller should adopt its case_id rather than assuming one.
 */
export async function saveCase(payload) {
  const response = await request("/api/cases", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    fallback: "Could not save the case",
  });

  const data = await response.json();

  return data.case;
}

export async function fetchCase(caseId) {
  const response = await request(
    `/api/cases/${encodeURIComponent(caseId)}`,
    { fallback: "Could not load that case" }
  );

  const data = await response.json();

  return data.case;
}

export async function deleteCase(caseId) {
  await request(`/api/cases/${encodeURIComponent(caseId)}`, {
    method: "DELETE",
    fallback: "Could not delete that case",
  });

  return true;
}

/**
 * Fetch a set of authenticated render endpoints as blob URLs, keyed the way
 * they arrived.
 *
 * Renders cannot be loaded with a plain <img src> pointing at the API: an
 * <img> tag cannot send an Authorization header, and putting the token in
 * the query string would write it into uvicorn's access log in plaintext.
 * So each one is fetched properly and wrapped in an object URL.
 *
 * The caller owns the returned URLs and MUST pass them to releaseImages()
 * when they are no longer displayed, or the blobs leak for the lifetime of
 * the page.
 */
async function fetchBlobMap(paths) {
  const entries = await Promise.all(
    Object.entries(paths).map(async ([key, path]) => {
      try {
        const response = await request(path, {
          fallback: `Could not load the ${key} render`,
        });

        return [key, URL.createObjectURL(await response.blob())];
      } catch (error) {
        // One missing render must not blank the whole case. A saved case can
        // legitimately lack the saliency images if attribution failed on the
        // run that produced it.
        if (error.status === 401) throw error;
        return null;
      }
    })
  );

  return Object.fromEntries(entries.filter(Boolean));
}

/**
 * Fetch a case's stored echo PNGs as blob URLs.
 *
 * @param {string} caseId
 * @param {string[]} names  render keys present on the case
 * @returns {Promise<Record<string, string>>} name -> blob URL
 */
export async function fetchCaseImages(caseId, names = []) {
  if (!caseId || names.length === 0) return {};

  return await fetchBlobMap(
    Object.fromEntries(
      names.map((name) => [
        name,
        `/api/cases/${encodeURIComponent(caseId)}` +
          `/images/${encodeURIComponent(name)}`,
      ])
    )
  );
}

/**
 * Fetch a case's stored ECG figures as blob URLs.
 *
 * Takes the record's `ecg_figures` map exactly as the backend sent it —
 * `{ strip: "/api/cases/…/images/ecg_strip" }` — and follows those paths
 * rather than rebuilding them from the keys. The short names the UI uses and
 * the storage keys the endpoint serves are deliberately different, and
 * duplicating that translation on this side would let the two drift apart
 * with nothing to catch it but a blank panel.
 *
 * @param {Record<string, string>} figures  short name -> API path
 * @returns {Promise<Record<string, string>>} short name -> blob URL
 */
export async function fetchCaseFigures(figures) {
  const paths = Object.fromEntries(
    Object.entries(figures || {}).filter(
      ([, path]) => typeof path === "string" && path.startsWith("/")
    )
  );

  if (Object.keys(paths).length === 0) return {};

  return await fetchBlobMap(paths);
}

/** Revoke blob URLs created by fetchCaseImages or fetchCaseFigures. */
export function releaseImages(images) {
  if (!images) return;

  for (const url of Object.values(images)) {
    if (typeof url === "string" && url.startsWith("blob:")) {
      URL.revokeObjectURL(url);
    }
  }
}
