/*
 * ============================================================
 * REPORT RESULT
 *
 * Renders the structured report returned by POST /api/report.
 *
 * The report exists to answer one question that no single
 * modality view can: what does this case amount to across
 * everything that was and was not analysed. That makes it the
 * screen most likely to be mistaken for a diagnosis, so three
 * rules shape it:
 *
 *   1. There is no fusion model. The integration is a
 *      deterministic aggregation, `learned_fusion: false`
 *      comes back in the payload, and it is printed rather
 *      than paraphrased.
 *
 *   2. A missing narrative is not a failed report. MedGemma
 *      contributes exactly one field. When it is absent the
 *      backend returns 200 with `ai_summary_error`, and this
 *      component renders every model finding intact beside a
 *      stated reason — never an error page.
 *
 *   3. Cross-modal observations are observations. Each one
 *      carries `inference: "none"` from the backend, and that
 *      is surfaced as a chip on the card, because two findings
 *      printed side by side is exactly where a causal reading
 *      gets invented.
 * ============================================================
 */

const MODALITY_ORDER = ["ccta", "echo", "ecg"];

/* Deliberately not colour-coded by "good" and "bad". `analysed` is the only
   status that means a model ran; the other three are different kinds of
   nothing, and flattening them into one grey chip is what makes an
   unanalysed modality read as a negative finding. */
const STATUS_LABEL = {
  analysed: "Analysed",
  not_provided: "No input",
  provided_not_analysed: "Not analysed",
  no_model: "No model",
};

const STATUS_TONE = {
  analysed: "ok",
  not_provided: "none",
  provided_not_analysed: "warn",
  no_model: "none",
};

const SEVERITY_ORDER = { warning: 0, note: 1 };

const KIND_LABEL = {
  model_limitation: "Model limitation",
  input_quality: "Input quality",
  coverage: "Coverage",
  contradiction: "Contradiction",
  pairing: "Pairing",
};

const RECOMMENDATION_LABEL = {
  analysis: "Run an analysis",
  input: "Missing input",
  capability: "Not available",
  coverage: "Incomplete coverage",
  interpretation: "Interpretation",
};

function formatTimestamp(value) {
  if (!value) return "—";

  const parsed = new Date(value);

  if (Number.isNaN(parsed.getTime())) return value;

  return parsed.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

/* A blank identity field is a fact about the record, not a rendering gap. */
function orNotRecorded(value) {
  return value === null || value === undefined || value === ""
    ? "Not recorded"
    : String(value);
}

function formatNumber(value, digits = 3) {
  return typeof value === "number" ? value.toFixed(digits) : "—";
}

/* Field names arrive as "clinical:age" so the prompt can namespace them. */
function fieldName(name) {
  return String(name).split(":").slice(-1)[0].replace(/_/g, " ");
}

function paragraphs(text) {
  return String(text)
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter(Boolean);
}

function ReportResult({ report, prompt = null }) {
  if (!report) return null;

  const patient = report.patient || {};
  const generatedBy = report.generated_by || {};
  const integrated = report.integrated_evidence || {};
  const integration = integrated.integration_method || {};
  const clinical = report.clinical_context || {};
  const modalities = report.modality_results || {};

  const available = integrated.available_modalities || [];
  const missing = integrated.missing_modalities || [];
  const observations = integrated.cross_modal_evidence || [];
  const recommendations = report.recommendations || [];
  const modelVersions = report.model_versions || {};

  /* `missing_modalities` can carry "clinical", which is not a modality — it is
     the operator's own form. Mixing it into a sentence about absent imaging
     would misdescribe both. */
  const missingModalities = missing.filter((key) => key !== "clinical");
  const clinicalMissing = missing.includes("clinical");

  /* Warnings before notes. Within a severity the backend's order is kept —
     it emits them per modality, which is the order a reader scans in. */
  const uncertainties = [...(report.uncertainties || [])].sort(
    (a, b) =>
      (SEVERITY_ORDER[a.severity] ?? 9) - (SEVERITY_ORDER[b.severity] ?? 9)
  );

  const warnings = uncertainties.filter((item) => item.severity === "warning");

  const summaryParagraphs = report.ai_summary
    ? paragraphs(report.ai_summary)
    : [];

  return (
    <div className="cv-report-result">
      {/* ====================================================
          HEADER
      ==================================================== */}

      <div className="cv-ecg-headline">
        <div>
          <span className="cv-card-kicker">Integrated clinical AI report</span>

          <h3>
            {orNotRecorded(patient.name)}
            {report.case_id ? ` — case ${report.case_id}` : " — unsaved case"}
          </h3>

          <p>
            Assembled by a <strong>deterministic software layer</strong> from
            the outputs of three independently trained models. There is{" "}
            <strong>no fusion model</strong> in this project: nothing below was
            learned jointly, no modality was weighted against another, and no
            combined risk was computed.
          </p>
        </div>

        <div className="cv-ecg-calls">
          <div
            className={`cv-ecg-call ${available.length ? "" : "none"}`}
          >
            {available.length} of {MODALITY_ORDER.length} analysed
            <em>
              {available.length
                ? available.join(", ")
                : "nothing has been run"}
            </em>
          </div>

          <div
            className={`cv-ecg-call ${warnings.length ? "weak" : "norm"}`}
          >
            {warnings.length
              ? `${warnings.length} warning${warnings.length > 1 ? "s" : ""}`
              : "No warnings"}
            <em>
              {uncertainties.length} uncertaint
              {uncertainties.length === 1 ? "y" : "ies"} recorded
            </em>
          </div>
        </div>
      </div>

      {/* ====================================================
          NARRATIVE SUMMARY

          First because it is what a reader looks for, and
          bounded immediately afterwards because it is the one
          part of this report that no model measured.
      ==================================================== */}

      <div className="cv-ecg-block">
        <span className="cv-card-kicker">Narrative summary</span>

        {summaryParagraphs.length > 0 ? (
          <>
            <div className="cv-report-summary">
              {summaryParagraphs.map((block, index) => (
                <p key={index}>{block}</p>
              ))}
            </div>

            <div className="cv-ecg-note">
              <strong>What this paragraph is.</strong>{" "}
              {report.ai_summary_scope}
            </div>

            {generatedBy.summary_model && (
              <div className="cv-metrics-note secondary">
                <strong>Written by {generatedBy.summary_model}</strong>

                <span>
                  A language model running locally. It read only the structured
                  evidence in this report — it never saw the images, the
                  waveform or the volume.
                </span>
              </div>
            )}
          </>
        ) : (
          /* Not an error state. Every finding in this report came from a
             modality model; the language model contributes one field. */
          <div className="cv-xai-unavailable">
            <strong>No narrative was written for this case</strong>

            <p>
              {report.ai_summary_error ||
                "The narrative summary is not available."}
            </p>

            <p>
              Everything else in this report is intact. The summary is the only
              part written by a language model, so its absence costs one
              paragraph and no findings.
            </p>
          </div>
        )}

        {prompt?.context && (
          /* Offered so a reader can check the narrative against exactly what
             the model was given, rather than trusting that it stayed inside
             the evidence. */
          <details className="cv-report-prompt">
            <summary>Show the exact text the language model was given</summary>

            <pre>{prompt.context}</pre>

            {prompt.question && (
              <>
                <strong>Question</strong>

                <pre>{prompt.question}</pre>
              </>
            )}
          </details>
        )}
      </div>

      {/* ====================================================
          MODALITY STATUS

          Four statuses, kept distinct. "No input", "not
          analysed" and "no model" are three different reasons
          for an empty result and only one of them is something
          the operator can fix.
      ==================================================== */}

      <div className="cv-ecg-block">
        <span className="cv-card-kicker">What each modality contributed</span>

        <div className="cv-report-modalities">
          {MODALITY_ORDER.map((key) => {
            const modality = modalities[key] || {};
            const status = modality.status || "not_provided";
            const findings = modality.findings || [];
            const coverage = modality.coverage || {};

            return (
              <div key={key} className="cv-report-modality">
                <div className="cv-report-modality-head">
                  <strong>{modality.label || key.toUpperCase()}</strong>

                  <span
                    className={`cv-report-status ${
                      STATUS_TONE[status] || "none"
                    }`}
                  >
                    {STATUS_LABEL[status] || status}
                  </span>
                </div>

                <p className="cv-report-modality-meaning">
                  {modality.status_meaning}
                </p>

                {modality.task && (
                  <p className="cv-report-modality-task">
                    {modality.model?.name
                      ? `${modality.model.name} — `
                      : ""}
                    {modality.task}
                  </p>
                )}

                {coverage.complete === false && (
                  <p className="cv-report-modality-warn">
                    Only {formatNumber(coverage.analysed_percent, 2)}% of this
                    input was examined. The rest is unanalysed, not negative.
                  </p>
                )}

                {findings.length > 0 && (
                  <ul className="cv-report-findings">
                    {findings.map((finding, index) => (
                      <li
                        key={index}
                        className={finding.observed ? "yes" : "no"}
                      >
                        <span>{finding.name || finding.label || "—"}</span>

                        <em>
                          {finding.observed ? "observed" : "not observed"}
                          {typeof finding.confidence?.probability === "number"
                            ? ` · p ${formatNumber(
                                finding.confidence.probability
                              )}`
                            : ""}
                          {typeof finding.measurement?.volume_ml === "number"
                            ? ` · ${finding.measurement.volume_ml.toFixed(
                                2
                              )} mL`
                            : ""}
                          {typeof finding.measurement?.area_cm2 === "number"
                            ? ` · ${finding.measurement.area_cm2.toFixed(
                                2
                              )} cm²`
                            : ""}
                        </em>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            );
          })}
        </div>

        {missingModalities.length > 0 && (
          <div className="cv-ecg-note">
            <strong>Not in this report:</strong>{" "}
            {missingModalities.join(", ")}. Absent modalities are absent
            evidence. They are not normal results, and nothing above should be
            read as covering them.
          </div>
        )}
      </div>

      {/* ====================================================
          CROSS-MODAL OBSERVATIONS
      ==================================================== */}

      <div className="cv-ecg-block">
        <span className="cv-card-kicker">Observed together</span>

        <p className="cv-ecg-block-intro">
          Findings that appeared in the same case.{" "}
          <strong>{integration.type}</strong> — learned fusion:{" "}
          <strong>
            {integration.learned_fusion === false ? "none" : "unknown"}
          </strong>
          . {integration.note}
        </p>

        {observations.length > 0 ? (
          <div className="cv-report-observations">
            {observations.map((item, index) => (
              <div key={index} className="cv-report-observation">
                <div className="cv-report-observation-head">
                  <strong>{KIND_LABEL[item.kind] || item.kind}</strong>

                  {/* Printed from the payload, not written here. It is the
                      field that stops a pair of findings being read as a
                      mechanism. */}
                  <span className="cv-report-inference">
                    inference: {item.inference}
                  </span>
                </div>

                <p>{item.statement}</p>

                <span className="cv-report-basis">
                  {(item.modalities || []).join(" + ")} — {item.basis}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="cv-muted-text">
            Nothing was observed across more than one modality in this case.
            With fewer than two modalities analysed there is nothing to place
            side by side.
          </p>
        )}
      </div>

      {/* ====================================================
          UNCERTAINTIES
      ==================================================== */}

      <div className="cv-ecg-block">
        <span className="cv-card-kicker">Uncertainties</span>

        <p className="cv-ecg-block-intro">
          Collected from the model cards, the inputs and the coverage of this
          particular run. This list is not a disclaimer — every entry is
          something specific about <strong>this case</strong>.
        </p>

        {uncertainties.length > 0 ? (
          <div className="cv-report-uncertainties">
            {uncertainties.map((item, index) => (
              <div
                key={index}
                className={
                  item.severity === "warning"
                    ? "cv-ecg-warning"
                    : "cv-ecg-note"
                }
              >
                <strong>
                  {item.scope} — {KIND_LABEL[item.kind] || item.kind}
                </strong>

                <span>{item.detail}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="cv-muted-text">
            No uncertainty was recorded, which happens when nothing has been
            analysed yet — not when a result is certain.
          </p>
        )}
      </div>

      {/* ====================================================
          CLINICAL CONTEXT
      ==================================================== */}

      <div className="cv-ecg-block">
        <span className="cv-card-kicker">Clinical context</span>

        <p className="cv-ecg-block-intro">
          {clinical.source || "Entered by the operator"}.{" "}
          {clinical.interpretation}
        </p>

        {/* Every field blank is its own state, and it changes how the rest of
            the report should be read: there is no presentation to read the
            model findings against. */}
        {clinicalMissing && (
          <div className="cv-ecg-warning">
            <strong>No clinical context was recorded for this case</strong>

            <span>
              Nothing in this report can be read against the patient&rsquo;s
              presentation. The model findings below stand alone — they were
              produced from the images and the waveform only.
            </span>
          </div>
        )}

        <div className="cv-model-grid">
          <div>
            <span>Age</span>

            <strong>
              {orNotRecorded(clinical.age)}
              {clinical.age && clinical.age_source ? (
                <em> ({clinical.age_source})</em>
              ) : null}
            </strong>
          </div>

          <div>
            <span>Sex</span>

            <strong>{orNotRecorded(clinical.sex)}</strong>
          </div>

          <div>
            <span>Blood pressure</span>

            <strong>{orNotRecorded(clinical.blood_pressure)}</strong>
          </div>

          <div>
            <span>Heart rate</span>

            <strong>{orNotRecorded(clinical.heart_rate)}</strong>
          </div>
        </div>

        {[
          ["Symptoms", clinical.symptoms],
          ["History", clinical.history],
          ["Medications", clinical.medications],
        ].map(([label, values]) => (
          <p key={label} className="cv-report-context-line">
            <strong>{label}:</strong>{" "}
            {values?.length ? values.join(", ") : "Not recorded"}
          </p>
        ))}

        {clinical.notes && (
          <p className="cv-report-context-line">
            <strong>Notes:</strong> {clinical.notes}
          </p>
        )}

        {clinical.unknown_fields?.length > 0 && (
          <div className="cv-ecg-note warn">
            <strong>Blank, not negative.</strong> These fields were never
            filled in:{" "}
            {clinical.unknown_fields.map(fieldName).join(", ")}. An unknown is
            weaker evidence than a recorded negative, and nothing in this
            report treats them as ruled out.
          </div>
        )}

        {clinical.not_collected_fields?.length > 0 && (
          <div className="cv-metrics-note secondary">
            <strong>Not collected by this application</strong>

            <span>
              {clinical.not_collected_fields.map(fieldName).join(", ")}. These
              are not part of any form in CardioVision, so their absence says
              nothing about the patient.
            </span>
          </div>
        )}
      </div>

      {/* ====================================================
          RECOMMENDATIONS

          Workflow steps. The scope line goes above the list,
          because a list of actions under a cardiology report
          reads as management advice unless it is told not to.
      ==================================================== */}

      <div className="cv-ecg-block">
        <span className="cv-card-kicker">Next steps</span>

        <div className="cv-ecg-note warn">
          <strong>These are workflow steps, not clinical advice.</strong>{" "}
          {report.recommendations_scope}
        </div>

        {recommendations.length > 0 ? (
          <ol className="cv-report-actions">
            {recommendations.map((item, index) => (
              <li key={index}>
                <span className="cv-report-action-kind">
                  {RECOMMENDATION_LABEL[item.kind] || item.kind}
                  {item.modality ? ` · ${item.modality}` : ""}
                </span>

                <strong>{item.action}</strong>

                <em>{item.reason}</em>
              </li>
            ))}
          </ol>
        ) : (
          <p className="cv-muted-text">No steps were generated.</p>
        )}
      </div>

      {/* ====================================================
          PROVENANCE
      ==================================================== */}

      <div className="cv-ecg-block">
        <span className="cv-card-kicker">Provenance</span>

        <div className="cv-model-grid">
          <div>
            <span>Generated</span>

            <strong>{formatTimestamp(report.generated_at)}</strong>
          </div>

          <div>
            <span>Schema version</span>

            <strong>{report.schema_version || "—"}</strong>
          </div>

          <div>
            <span>Application</span>

            <strong>
              {generatedBy.application || "CardioVision AI"}
              {generatedBy.version ? ` ${generatedBy.version}` : ""}
            </strong>
          </div>

          <div>
            <span>Integration</span>

            <strong>{generatedBy.evidence_layer || integration.type}</strong>
          </div>

          <div>
            <span>Patient ID</span>

            <strong>{orNotRecorded(patient.patient_id)}</strong>
          </div>

          <div>
            <span>Sex on record</span>

            <strong>{orNotRecorded(patient.sex)}</strong>
          </div>

          <div>
            <span>Date of birth</span>

            <strong>{orNotRecorded(patient.date_of_birth)}</strong>
          </div>

          <div>
            <span>Study date</span>

            <strong>{orNotRecorded(patient.study_date)}</strong>
          </div>
        </div>

        {Object.keys(modelVersions).length > 0 && (
          <>
            <p className="cv-ecg-block-intro">
              <strong>Models that produced the findings above.</strong> Each
              ran independently; none saw another&rsquo;s output. The{" "}
              <code>fusion</code> row is listed with the rest precisely because
              it has no model behind it.
            </p>

            <ul className="cv-report-versions">
              {Object.entries(modelVersions).map(([key, value]) => {
                const entry = value && typeof value === "object" ? value : {};

                return (
                  <li key={key}>
                    <span>{key}</span>

                    {/* `model: null` is the fusion layer, and it is spelled
                        out rather than left blank. */}
                    <strong>{entry.model || "No model"}</strong>

                    <em>
                      {[entry.task, entry.dataset, entry.note]
                        .filter(Boolean)
                        .join(" · ")}
                    </em>
                  </li>
                );
              })}
            </ul>
          </>
        )}
      </div>

      {/* ====================================================
          DISCLAIMER
      ==================================================== */}

      <div className="cv-echo-disclaimer">{report.disclaimer}</div>
    </div>
  );
}

export default ReportResult;
