import { useEffect, useState } from "react";

/*
 * ============================================================
 * CASE LIST
 *
 * Saved cases, most recently updated first. Lives in the sidebar so
 * switching between patients never requires navigating away from the
 * current one.
 *
 * Search is debounced and runs on the backend rather than filtering a
 * cached array, so it still finds cases beyond the fetch limit.
 * ============================================================
 */

function formatWhen(iso) {
  if (!iso) return "";

  const when = new Date(iso);

  if (Number.isNaN(when.getTime())) return "";

  const now = new Date();
  const sameDay = when.toDateString() === now.toDateString();

  if (sameDay) {
    return when.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  const thisYear = when.getFullYear() === now.getFullYear();

  return when.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    ...(thisYear ? {} : { year: "numeric" }),
  });
}

function CaseList({
  cases,
  total,
  activeCaseId,
  isLoading,
  error,
  onSearch,
  onOpen,
  onDelete,
  onRefresh,
}) {
  const [term, setTerm] = useState("");
  const [confirmingDelete, setConfirmingDelete] = useState(null);

  // Debounced so typing a name does not fire a request per keystroke.
  useEffect(() => {
    const timer = setTimeout(() => onSearch(term), 250);

    return () => clearTimeout(timer);
  }, [term, onSearch]);

  // A case deleted or opened elsewhere should not leave a stale confirm
  // button armed on a row that no longer means what it did.
  useEffect(() => {
    setConfirmingDelete(null);
  }, [cases]);

  const searching = term.trim().length > 0;

  return (
    <div className="cv-case-list">
      <div className="cv-case-list-head">
        <span className="cv-sidebar-label">Saved cases</span>

        <span className="cv-case-list-count">{total}</span>
      </div>

      <input
        type="search"
        className="cv-case-search"
        placeholder="Search name, MRN, case ID"
        value={term}
        onChange={(event) => setTerm(event.target.value)}
      />

      {error && (
        <div className="cv-case-list-error">
          <span>{error}</span>

          <button type="button" onClick={onRefresh}>
            Retry
          </button>
        </div>
      )}

      {!error && isLoading && cases.length === 0 && (
        <div className="cv-case-list-empty">Loading…</div>
      )}

      {!error && !isLoading && cases.length === 0 && (
        <div className="cv-case-list-empty">
          {searching
            ? `Nothing matches “${term.trim()}”.`
            : "No saved cases yet. Fill in a patient and press Save case."}
        </div>
      )}

      <div className="cv-case-items">
        {cases.map((item) => {
          const isActive = item.case_id === activeCaseId;
          const isConfirming = confirmingDelete === item.case_id;

          return (
            <div
              key={item.case_id}
              className={`cv-case-item ${isActive ? "active" : ""}`}
            >
              <button
                type="button"
                className="cv-case-item-open"
                onClick={() => onOpen(item.case_id)}
                title={`Open ${item.display_name}`}
              >
                <span className="cv-case-item-name">{item.display_name}</span>

                <span className="cv-case-item-meta">
                  {item.patient_mrn && (
                    <span className="cv-case-item-mrn">{item.patient_mrn}</span>
                  )}

                  <span>{formatWhen(item.updated_at)}</span>
                </span>

                <span className="cv-case-item-tags">
                  {item.echo_analyzed ? (
                    <span className="cv-case-tag echo">
                      Echo · {item.structures_found}/3
                    </span>
                  ) : (
                    <span className="cv-case-tag empty">No imaging</span>
                  )}
                </span>
              </button>

              {isConfirming ? (
                <div className="cv-case-item-confirm">
                  <button
                    type="button"
                    className="cv-case-item-danger"
                    onClick={() => {
                      onDelete(item.case_id);
                      setConfirmingDelete(null);
                    }}
                  >
                    Delete
                  </button>

                  <button
                    type="button"
                    onClick={() => setConfirmingDelete(null)}
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  className="cv-case-item-remove"
                  aria-label={`Delete ${item.display_name}`}
                  title="Delete this case"
                  onClick={() => setConfirmingDelete(item.case_id)}
                >
                  ×
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default CaseList;
