/*
 * ============================================================
 * PENDING MODEL
 *
 * Shown for modalities that have no trained model yet. Deliberately
 * states what is missing instead of displaying placeholder metrics —
 * an invented confidence score is indistinguishable from a real one
 * once it is on screen.
 * ============================================================
 */

function PendingModel({ label, note, file, requirement }) {
  return (
    <div className="cv-pending">
      <div className="cv-pending-badge">Model not yet trained</div>

      <h3>{label}</h3>

      <p>{note}</p>

      {file && (
        <div className="cv-pending-file">
          <strong>{file.name}</strong>

          <span>
            Uploaded and held in this case, but not analysed — there is no
            model to run it through yet.
          </span>
        </div>
      )}

      {requirement && (
        <div className="cv-pending-requirement">
          <span className="cv-card-kicker">Needed to enable this</span>

          <p>{requirement}</p>
        </div>
      )}

      <div className="cv-pending-note">
        No metrics are shown here on purpose. Nothing about this modality
        has been measured, so any number would be fabricated.
      </div>
    </div>
  );
}

export default PendingModel;
