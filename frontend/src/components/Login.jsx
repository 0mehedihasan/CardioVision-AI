import { useEffect, useRef, useState } from "react";

/*
 * ============================================================
 * LOGIN
 *
 * Gates the whole application. The credential check happens on the
 * backend — this screen only collects them and reports what came back.
 *
 * Backend and model state are shown before sign-in, because "my password
 * is not working" and "the backend is not running" look identical from
 * behind a login form otherwise.
 * ============================================================
 */

function Login({ health, healthError, onRetryHealth, onSignIn }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const usernameRef = useRef(null);

  useEffect(() => {
    usernameRef.current?.focus();
  }, []);

  const backendOnline = Boolean(health);
  const echoReady = Boolean(health?.modalities?.echo?.available);
  const medgemmaReady = Boolean(health?.models?.medgemma?.loaded);
  const storageReady = Boolean(health?.storage?.ready);
  const savedCases = health?.storage?.saved_cases ?? 0;

  const submit = async (event) => {
    event.preventDefault();

    if (isSubmitting) return;

    if (!username.trim() || !password) {
      setError("Enter both a username and a password.");
      return;
    }

    setError(null);
    setIsSubmitting(true);

    try {
      await onSignIn(username.trim(), password);
      // On success the parent unmounts this component, so there is nothing
      // to reset here.
    } catch (signInError) {
      setError(signInError.message);
      // Clear only the password. Retyping a correct username after a typo in
      // the password is pure friction.
      setPassword("");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="cv-login">
      <div className="cv-login-panel">
        <div className="cv-login-brand">
          <div className="cv-brand-mark">CV</div>

          <div>
            <div className="cv-brand-name">CardioVision</div>

            <div className="cv-brand-subtitle">
              Multimodal cardiovascular intelligence
            </div>
          </div>
        </div>

        <div className="cv-login-intro">
          <h1>Sign in</h1>

          <p>
            This workstation runs entirely on your own machine. Patient
            records, images and questions never leave it.
          </p>
        </div>

        <form className="cv-login-form" onSubmit={submit}>
          <label>
            Username

            <input
              ref={usernameRef}
              type="text"
              autoComplete="username"
              spellCheck="false"
              autoCapitalize="none"
              value={username}
              disabled={isSubmitting}
              onChange={(event) => setUsername(event.target.value)}
            />
          </label>

          <label>
            Password

            <input
              type="password"
              autoComplete="current-password"
              value={password}
              disabled={isSubmitting}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>

          {error && (
            <div className="cv-login-error" role="alert">
              {error}
            </div>
          )}

          <button
            type="submit"
            className={`cv-primary-button ${isSubmitting ? "loading" : ""}`}
            disabled={isSubmitting || !backendOnline}
          >
            {isSubmitting ? (
              <>
                <span className="cv-button-spinner" />
                Signing in
              </>
            ) : (
              <>
                Sign in
                <span aria-hidden="true">→</span>
              </>
            )}
          </button>
        </form>

        {/* ==================================================
            BACKEND STATE

            Shown before sign-in so a stopped backend is not
            mistaken for a rejected password.
        ================================================== */}

        <div className="cv-login-status">
          {backendOnline ? (
            <>
              <StatusLine
                label="Backend"
                value={`Online · ${(health.device || "unknown").toUpperCase()}`}
                ok
              />

              <StatusLine
                label="Echo segmentation"
                value={echoReady ? "Model loaded" : "Unavailable"}
                ok={echoReady}
              />

              <StatusLine
                label="Clinical language model"
                value={medgemmaReady ? "Loaded" : "Unavailable"}
                ok={medgemmaReady}
              />

              <StatusLine
                label="Case storage"
                value={
                  storageReady
                    ? `Ready · ${savedCases} saved ${
                        savedCases === 1 ? "case" : "cases"
                      }`
                    : "Unavailable"
                }
                ok={storageReady}
              />
            </>
          ) : (
            <div className="cv-login-offline">
              <strong>Backend offline</strong>

              <span>{healthError}</span>

              <button
                type="button"
                className="cv-secondary-button"
                onClick={onRetryHealth}
              >
                Retry
              </button>
            </div>
          )}
        </div>

        {/* An unchanged default password is worth saying out loud every
            time, not once in a README nobody reopens. */}
        {health?.auth?.using_default_credentials && (
          <div className="cv-login-note">
            <strong>Default password in use</strong>

            <span>
              This is an access gate, not security. Set{" "}
              <code>CARDIOVISION_USER</code> and{" "}
              <code>CARDIOVISION_PASSWORD</code> before starting the backend
              to change it, and keep the service bound to localhost.
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function StatusLine({ label, value, ok }) {
  return (
    <div className={`cv-status-line ${ok ? "ok" : "warn"}`}>
      <span className="cv-status-line-dot" />

      <span className="cv-status-line-label">{label}</span>

      <span className="cv-status-line-value">{value}</span>
    </div>
  );
}

export default Login;
