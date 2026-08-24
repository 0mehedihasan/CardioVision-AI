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

export const API_BASE_URL = "http://127.0.0.1:8000";

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

function offlineMessage() {
  return (
    "Cannot reach the CardioVision backend at " +
    `${API_BASE_URL}. Start it with: ` +
    "uvicorn main:app --reload --port 8000 (from the backend/ directory)."
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
 * Fetch a case's stored PNGs as blob URLs.
 *
 * Images cannot be loaded with a plain <img src> pointing at the API: an
 * <img> tag cannot send an Authorization header, and putting the token in
 * the query string would write it into uvicorn's access log in plaintext.
 * So each image is fetched properly and wrapped in an object URL.
 *
 * The caller owns the returned URLs and MUST pass them to releaseImages()
 * when they are no longer displayed, or the blobs leak for the lifetime of
 * the page.
 *
 * @param {string} caseId
 * @param {string[]} names  render keys present on the case
 * @returns {Promise<Record<string, string>>} name -> blob URL
 */
export async function fetchCaseImages(caseId, names = []) {
  if (!caseId || names.length === 0) return {};

  const entries = await Promise.all(
    names.map(async (name) => {
      try {
        const response = await request(
          `/api/cases/${encodeURIComponent(caseId)}` +
            `/images/${encodeURIComponent(name)}`,
          { fallback: `Could not load the ${name} image` }
        );

        const blob = await response.blob();

        return [name, URL.createObjectURL(blob)];
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

/** Revoke blob URLs created by fetchCaseImages. */
export function releaseImages(images) {
  if (!images) return;

  for (const url of Object.values(images)) {
    if (typeof url === "string" && url.startsWith("blob:")) {
      URL.revokeObjectURL(url);
    }
  }
}
