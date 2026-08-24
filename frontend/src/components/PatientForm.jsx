/*
 * ============================================================
 * PATIENT FORM
 *
 * Identity and study metadata for a case. Separate from the clinical
 * form because these fields answer "who and when", not "what is wrong" —
 * and because the name is deliberately withheld from the language-model
 * prompt while the clinical fields are sent.
 * ============================================================
 */

/** Age from a date of birth, mirroring the backend's derive_age(). */
function deriveAge(dateOfBirth) {
  if (!dateOfBirth) return null;

  const born = new Date(`${dateOfBirth}T00:00:00`);

  if (Number.isNaN(born.getTime())) return null;

  const now = new Date();

  if (born > now) return null;

  let age = now.getFullYear() - born.getFullYear();

  // Subtract a year when this year's birthday has not happened yet.
  const monthDiff = now.getMonth() - born.getMonth();

  if (monthDiff < 0 || (monthDiff === 0 && now.getDate() < born.getDate())) {
    age -= 1;
  }

  return age >= 0 && age < 150 ? age : null;
}

function PatientForm({ data, onChange, caseId, isSaved, savedAt }) {
  const age = deriveAge(data.dateOfBirth);
  const today = new Date().toISOString().slice(0, 10);

  return (
    <div className="cv-patient-card">
      <div className="cv-panel-header">
        <span className="cv-card-kicker">Patient record</span>

        <h3>Identity and study</h3>
      </div>

      <div className="cv-form">
        <label>
          Patient name

          <input
            type="text"
            placeholder="Family name, given name"
            value={data.name}
            onChange={(event) => onChange("name", event.target.value)}
          />
        </label>

        <div className="cv-form-row">
          <label>
            Medical record number

            <input
              type="text"
              placeholder="e.g. MRN-48213"
              spellCheck="false"
              value={data.mrn}
              onChange={(event) => onChange("mrn", event.target.value)}
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
              <option value="Other">Other</option>
            </select>
          </label>
        </div>

        <div className="cv-form-row">
          <label>
            Date of birth
            {/* Shown live so a mistyped year is obvious before saving. */}
            {age !== null && <em className="cv-field-derived">{age} years</em>}

            <input
              type="date"
              max={today}
              value={data.dateOfBirth}
              onChange={(event) => onChange("dateOfBirth", event.target.value)}
            />
          </label>

          <label>
            Study date

            <input
              type="date"
              value={data.studyDate}
              onChange={(event) => onChange("studyDate", event.target.value)}
            />
          </label>
        </div>

        <label>
          Referring clinician

          <input
            type="text"
            placeholder="Requesting physician or department"
            value={data.referringClinician}
            onChange={(event) =>
              onChange("referringClinician", event.target.value)
            }
          />
        </label>

        <label>
          Case notes

          <textarea
            rows="2"
            placeholder="Indication for the study, prior history, anything worth keeping with the record..."
            value={data.notes}
            onChange={(event) => onChange("notes", event.target.value)}
          />
        </label>

        <div className="cv-patient-footer">
          <div className="cv-patient-id">
            <span>Case ID</span>

            <strong>{caseId || "not assigned"}</strong>
          </div>

          <div className={`cv-patient-saved ${isSaved ? "saved" : ""}`}>
            {isSaved ? `Saved ${savedAt}` : "Unsaved"}
          </div>
        </div>

        <div className="cv-form-note">
          Stored locally in <code>data/cardiovision.db</code> on this machine.
          The name and MRN are kept out of the prompts sent to the language
          model; age, sex and notes are included.
        </div>
      </div>
    </div>
  );
}

export default PatientForm;
