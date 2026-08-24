/*
 * ============================================================
 * CardioVision AI — backend client
 * ============================================================
 */

export const API_BASE_URL = "http://127.0.0.1:8000";

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
   HEALTH
   ============================================================ */

export async function fetchHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);

    if (!response.ok) {
      throw new Error(await readError(response, "Health check failed"));
    }

    return await response.json();
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(offlineMessage());
    }
    throw error;
  }
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
 */
export async function analyzeEcho(file, options = {}) {
  const { frame, rotate = 0, flip = false, includeMask = true } = options;

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

  try {
    const response = await fetch(
      `${API_BASE_URL}/api/analyze/echo?${params.toString()}`,
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      throw new Error(
        await readError(response, "Echo segmentation failed")
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(offlineMessage());
    }
    throw error;
  }
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
  try {
    const response = await fetch(`${API_BASE_URL}/api/clinical-question`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        case: caseState,
      }),
    });

    if (!response.ok) {
      throw new Error(
        await readError(response, "Clinical question failed")
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(offlineMessage());
    }
    throw error;
  }
}
