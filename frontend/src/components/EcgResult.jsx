/*
 * ============================================================
 * ECG RESULT
 *
 * Displays the real output of the trained ECGResNet1D classifier over
 * the five PTB-XL diagnostic superclasses.
 *
 * Two things about this model shape the whole layout.
 *
 * The five outputs are INDEPENDENT sigmoids, not a softmax. They do not
 * sum to 1, a recording can carry none of them or several at once, and
 * NORM is one of the five rather than the absence of the other four. So
 * every probability is shown, always, with the threshold drawn on the
 * same axis — there is no single "the answer" row to highlight.
 *
 * And the macro numbers hide a weak class. Macro AUROC 0.913 reads as one
 * reassuring figure while hypertrophy sits at precision 0.361, meaning
 * roughly two in three positive HYP calls are false. Every positive call
 * therefore carries the precision it was measured at, right next to it,
 * and a weak class that was actually called gets a warning of its own.
 * ============================================================
 */

function formatProbability(value) {
  return typeof value === "number" ? value.toFixed(3) : "—";
}

function formatMetric(value, digits = 3) {
  return typeof value === "number" ? value.toFixed(digits) : "—";
}

/** Precision as the plain-language question a reader is actually asking. */
function falseCallRate(precision) {
  if (typeof precision !== "number" || precision <= 0) return null;

  return Math.round((1 - precision) * 100);
}

function EcgResult({ result }) {
  const model = result.model || {};
  const metrics = model.metrics || {};
  const input = result.input || {};
  const preprocessing = result.preprocessing || {};
  const figures = result.figures || {};
  const explainability = model.explainability || {};

  const threshold =
    typeof result.threshold === "number" ? result.threshold : 0.5;

  const predictions = result.predictions || [];
  const positives = result.positive_classes || [];
  const weakWarnings = result.weak_class_warnings || {};
  const weakClasses = metrics.weak_classes || {};

  const saliencyAvailable = Boolean(result.saliency_available);
  const leads = result.lead_attribution || [];

  // Sorted for reading, not re-ranked: this is the order a reader scans in,
  // and the checkpoint's column order carries no clinical meaning.
  const ranked = [...predictions].sort(
    (a, b) => (b.probability ?? 0) - (a.probability ?? 0)
  );

  const analysed = predictions.length > 0;

  if (!analysed) {
    return (
      <div className="cv-empty-analysis">
        <p>No ECG was classified in this case.</p>
      </div>
    );
  }

  return (
    <div className="cv-ecg-result">
      {/* ====================================================
          HEADLINE

          What was called, or an explicit statement that nothing
          was — which is a result, not a missing one.
      ==================================================== */}

      <div className="cv-ecg-headline">
        <div>
          <span className="cv-card-kicker">Screening result</span>

          <h3>
            {positives.length === 0
              ? "No superclass reached the calling threshold"
              : `${positives.length} of ${predictions.length} superclasses called positive`}
          </h3>

          {positives.length === 0 ? (
            /* The trap this paragraph exists to close: an empty result set
               looks like a clean bill of health. It is not one. NORM is
               itself one of the five classes and it was not called either,
               so the honest reading is "none of these five", not "normal". */
            <p>
              Every class fell below p ≥ {threshold}. That is a real result and
              it is <strong>not the same as a normal ECG</strong> — “Normal ECG”
              is one of the five classes and it was not called either. Read it
              as “none of these five patterns”, and read the tracing.
            </p>
          ) : (
            <p>
              Independent sigmoids, so these are five separate yes/no
              questions rather than one choice between five answers. The
              probabilities do not sum to 1 and more than one can be positive
              at once.
            </p>
          )}
        </div>

        <div className="cv-ecg-calls">
          {positives.length === 0 ? (
            <span className="cv-ecg-call none">None called</span>
          ) : (
            positives.map((name) => (
              <span
                key={name}
                className={`cv-ecg-call ${
                  name in weakClasses ? "weak" : ""
                } ${name === "NORM" ? "norm" : ""}`}
              >
                {model.class_labels?.[name] || name}

                <em>{name}</em>
              </span>
            ))
          )}
        </div>
      </div>

      {/* A contradiction the model can produce and cannot resolve: NORM is
          not the complement of the other four, so nothing stops it firing
          alongside them. Surfacing it is the only honest move — silently
          dropping one of the two calls would be inventing a decision. */}
      {positives.includes("NORM") && positives.length > 1 && (
        <div className="cv-ecg-contradiction">
          <strong>These calls contradict each other</strong>

          <span>
            “Normal ECG” was called positive alongside{" "}
            {positives
              .filter((name) => name !== "NORM")
              .map((name) => model.class_labels?.[name] || name)
              .join(", ")}
            . Because the five outputs are independent, nothing in the model
            prevents this, and the model has no way to say which call it
            prefers. Both are reported as-is.
          </span>
        </div>
      )}

      {/* ====================================================
          WEAK-CLASS WARNINGS

          Only for classes this recording actually called. A standing
          paragraph on every reading is the kind of boilerplate that
          stops being read, and the one time it mattered would look
          exactly like all the times it did not.
      ==================================================== */}

      {Object.entries(weakWarnings).map(([name, text]) => (
        <div key={name} className="cv-ecg-warning">
          <strong>
            {model.class_labels?.[name] || name} was called positive — this is
            the model's weakest class
          </strong>

          <span>{text}</span>
        </div>
      ))}

      {/* ====================================================
          THE STRIP
      ==================================================== */}

      {figures.strip ? (
        <div className="cv-ecg-strip">
          <div className="cv-ecg-strip-frame">
            <img
              src={figures.strip}
              alt="12-lead ECG strip with per-lead saliency"
            />
          </div>

          <div className="cv-echo-caption">
            <strong>
              12-lead strip
              {input.record_name ? ` · ${input.record_name}` : ""}
            </strong>

            <span>
              Plotted at 25 mm/s on standard paper gridding, from the filtered
              and resampled signal the model saw — before normalisation, so the
              trace stays in millivolts. The band under each lead is the
              gradient magnitude for{" "}
              {result.saliency_class
                ? model.class_labels?.[result.saliency_class] ||
                  result.saliency_class
                : "the reported class"}
              : darker means the logit was more sensitive to those samples.
            </span>
          </div>
        </div>
      ) : (
        <div className="cv-ecg-note">
          No waveform figure was rendered for this run, so there is no strip to
          show. The classification below is unaffected.
        </div>
      )}

      {/* ====================================================
          PROBABILITIES
      ==================================================== */}

      <div className="cv-ecg-block">
        <span className="cv-card-kicker">All five superclass probabilities</span>

        <div className="cv-ecg-predictions">
          {ranked.map((prediction) => (
            <PredictionRow
              key={prediction.name}
              prediction={prediction}
              threshold={threshold}
              weak={prediction.name in weakClasses}
            />
          ))}
        </div>

        <div className="cv-threshold-note">
          <strong>How “called positive” is decided</strong>

          <span>
            {result.threshold_note ||
              `A class is called positive at p ≥ ${threshold}.`}
          </span>
        </div>
      </div>

      {/* ====================================================
          LEAD ATTRIBUTION
      ==================================================== */}

      <div className="cv-ecg-block">
        <span className="cv-card-kicker">Lead attribution</span>

        {saliencyAvailable ? (
          <>
            <p className="cv-ecg-block-intro">
              Per-lead gradient magnitude for{" "}
              <strong>
                {model.class_labels?.[result.saliency_class] ||
                  result.saliency_class ||
                  "the reported class"}
              </strong>
              , scaled against the strongest lead. This shows{" "}
              <strong>where the model looked</strong>, not where an abnormality
              is — a high-ranking lead is not a finding in that lead, and a
              low-ranking one does not clear it.
            </p>

            {figures.lead_attribution && (
              <div className="cv-ecg-chart-frame">
                <img
                  src={figures.lead_attribution}
                  alt="Per-lead attribution ranking"
                />
              </div>
            )}

            <div className="cv-ecg-leads">
              {leads.map((lead, index) => (
                <div className="cv-ecg-lead-row" key={lead.name}>
                  <span className="cv-ecg-lead-rank">{index + 1}</span>

                  <span className="cv-ecg-lead-name">{lead.name}</span>

                  <div className="cv-perclass-bar">
                    <i style={{ width: `${(lead.score ?? 0) * 100}%` }} />
                  </div>

                  <strong>{formatMetric(lead.score, 3)}</strong>
                </div>
              ))}
            </div>

            {explainability.method && (
              <div className="cv-ecg-note">
                <strong>{explainability.method}</strong> — {explainability.note}
              </div>
            )}
          </>
        ) : (
          /* A zeroed gradient still renders as a perfectly plausible ranking
             of twelve leads. Showing it would be an explanation of a
             computation that never happened. */
          <div className="cv-xai-unavailable">
            <strong>Attribution unavailable for this run</strong>

            <p>
              The input gradient could not be computed, so there is no per-lead
              attribution and none is shown. The probabilities above are
              unaffected. Do not infer lead involvement from anything on this
              page.
            </p>
          </div>
        )}
      </div>

      {/* ====================================================
          MODEL CARD
      ==================================================== */}

      <div className="cv-ecg-block">
        <span className="cv-card-kicker">Model</span>

        <div className="cv-model-grid">
          <div>
            <span>Architecture</span>
            <strong>{model.architecture || "—"}</strong>
          </div>

          <div>
            <span>Task</span>
            <strong>{model.task || "—"}</strong>
          </div>

          <div>
            <span>Input</span>
            <strong>
              {model.input?.leads} leads × {model.input?.samples} samples @{" "}
              {model.input?.sampling_rate_hz} Hz
            </strong>
          </div>

          <div>
            <span>Parameters</span>
            <strong>
              {model.parameters ? model.parameters.toLocaleString() : "—"}
            </strong>
          </div>

          <div>
            <span>Best epoch</span>
            <strong>{model.best_epoch ?? "—"}</strong>
          </div>

          <div>
            <span>Calling threshold</span>
            <strong>p ≥ {model.threshold ?? threshold}</strong>
          </div>
        </div>

        <div className="cv-metrics-note">
          <strong>Held-out test performance</strong>

          <span>
            Measured on {metrics.test_records} records from{" "}
            {metrics.test_patients} patients in {metrics.dataset}, split at
            patient level. These describe the model overall — they are{" "}
            <strong>not</strong> a confidence score for this recording.
            {metrics.read_from_checkpoint
              ? " Read back from the checkpoint itself."
              : ""}
          </span>
        </div>

        <div className="cv-metric-grid">
          <div className="cv-metric">
            <span>Macro AUROC</span>
            <strong>{formatMetric(metrics.macro_AUROC, 4)}</strong>
          </div>

          <div className="cv-metric">
            <span>Macro AP</span>
            <strong>{formatMetric(metrics.macro_AP, 4)}</strong>
          </div>

          <div className="cv-metric">
            <span>Macro F1</span>
            <strong>{formatMetric(metrics.macro_F1, 4)}</strong>
          </div>

          <div className="cv-metric">
            <span>Macro precision</span>
            <strong>{formatMetric(metrics.macro_Precision, 4)}</strong>
          </div>
        </div>

        {/* Kept visually apart from the test block for the same reason as the
            echo card: the validation split selected the best epoch, so it is
            not an independent estimate of anything. */}
        <div className="cv-metrics-note secondary">
          <strong>Validation split (used during training)</strong>

          <span>
            This split chose the checkpoint, so it is not an independent
            estimate of performance — the test figures above are.
          </span>
        </div>

        <div className="cv-metric-grid">
          <div className="cv-metric muted">
            <span>Validation macro AUROC</span>
            <strong>{formatMetric(metrics.validation_macro_AUROC, 4)}</strong>
          </div>
        </div>

        {metrics.per_class && (
          <div className="cv-perclass">
            <span className="cv-card-kicker">
              Per-class test performance, all five
            </span>

            {/* The whole point of showing this table in full: the macro row
                above averages 0.36 precision together with 0.71 and reports
                one number. Averaged away, HYP is invisible. */}
            <table className="cv-ecg-metric-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>AUROC</th>
                  <th>AP</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>Prevalence</th>
                </tr>
              </thead>

              <tbody>
                {(model.class_names || Object.keys(metrics.per_class)).map(
                  (name) => {
                    const row = metrics.per_class[name] || {};

                    return (
                      <tr
                        key={name}
                        className={name in weakClasses ? "weak" : ""}
                      >
                        <td>
                          {row.label || name}

                          {name in weakClasses && (
                            <span className="cv-inline-badge warn">Weak</span>
                          )}
                        </td>

                        <td>{formatMetric(row.AUROC)}</td>
                        <td>{formatMetric(row.AP)}</td>
                        <td>{formatMetric(row.Precision)}</td>
                        <td>{formatMetric(row.Recall)}</td>
                        <td>{formatMetric(row.test_prevalence)}</td>
                      </tr>
                    );
                  }
                )}
              </tbody>
            </table>

            {Object.entries(weakClasses).map(([name, text]) => (
              <div key={name} className="cv-ecg-note warn">
                <strong>{model.class_labels?.[name] || name}</strong> — {text}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ====================================================
          INPUT PROVENANCE
      ==================================================== */}

      <div className="cv-ecg-block">
        <span className="cv-card-kicker">Recording</span>

        <div className="cv-model-grid">
          <div>
            <span>File</span>
            <strong title={input.filename}>{input.filename || "—"}</strong>
          </div>

          <div>
            <span>Format</span>
            <strong>{(input.format || "").toUpperCase() || "—"}</strong>
          </div>

          {input.companions?.length > 0 && (
            <div>
              <span>Companion files</span>
              <strong>{input.companions.join(", ")}</strong>
            </div>
          )}

          <div>
            <span>Sampling rate</span>
            <strong>
              {input.sampling_frequency_hz ?? "—"} Hz
              {input.resampled_to_hz &&
              input.resampled_to_hz !== input.sampling_frequency_hz
                ? ` → ${input.resampled_to_hz} Hz`
                : ""}
            </strong>
          </div>

          <div>
            <span>Samples</span>
            <strong>
              {input.samples_original ?? "—"} → {input.samples_analysed ?? "—"}
            </strong>
          </div>

          <div>
            <span>Duration</span>
            <strong>
              {typeof input.duration_seconds === "number"
                ? `${input.duration_seconds.toFixed(1)} s`
                : "—"}
            </strong>
          </div>

          <div>
            <span>Units</span>
            <strong>{input.units || "Not declared"}</strong>
          </div>

          <div>
            <span>Inference</span>
            <strong>
              {result.inference_ms} ms on {result.device}
              {result.configured_device &&
                result.device !== result.configured_device && (
                  <em> (fell back from {result.configured_device})</em>
                )}
            </strong>
          </div>
        </div>

        {/* Lead order is the failure that produces a confident wrong answer
            rather than an error: twelve columns in the wrong order still
            classify, they just classify a different heart. */}
        {input.lead_order_matches_training === false && (
          <div className="cv-ecg-warning">
            <strong>Lead order does not match the training layout</strong>

            <span>
              This recording's leads are{" "}
              <code>{(input.lead_names || []).join(", ")}</code>, which is not
              the order the model was trained on (I, II, III, aVR, aVL, aVF,
              V1–V6). The channels were used in the order supplied, so every
              probability above should be treated as unreliable until the
              layout is corrected.
            </span>
          </div>
        )}

        {!input.units && (
          <div className="cv-ecg-note">
            The source declared no units, so the strip's millivolt axis is an
            assumption rather than a calibration. It does not affect the
            classification — the model's input is normalised per lead — but do
            not read amplitudes off the figure.
          </div>
        )}

        <div className="cv-ecg-note">
          <strong>Preprocessing</strong> — bandpass{" "}
          {(preprocessing.bandpass_hz || []).join("–")} Hz, resampled to{" "}
          {preprocessing.resampled_to_hz} Hz, {preprocessing.length_samples}{" "}
          samples ({preprocessing.length_seconds} s), normalised{" "}
          {preprocessing.normalization}. {preprocessing.note}
        </div>

        {/* Mechanism, and the one place on this page where the HYP weakness
            has an explanation rather than just a number. */}
        <div className="cv-ecg-note">
          Because normalisation is per-lead, absolute voltage is removed before
          the model sees anything. It therefore cannot apply the millimetre
          voltage criteria a human reader uses for hypertrophy — which is
          consistent with hypertrophy being its weakest class — and no
          amplitude in millivolts can be attributed to it.
        </div>

        {result.notes?.length > 0 && (
          <ul className="cv-echo-notes">
            {result.notes.map((note, index) => (
              <li key={index}>{note}</li>
            ))}
          </ul>
        )}
      </div>

      <div className="cv-echo-disclaimer">
        Screening over five broad superclasses only. This model does not
        measure heart rate, rhythm, PR/QRS/QT intervals or axis; it does not
        localise an infarct or separate acute from old; it does not detect
        atrial fibrillation, which is not one of its classes; and it does not
        produce a diagnosis.
      </div>
    </div>
  );
}

/*
 * ============================================================
 * PREDICTION ROW
 *
 * One class. The bar carries a threshold marker at the calling point so
 * a near-miss reads as a near-miss rather than as a negative.
 *
 * The operating point is attached only to positive calls. Precision is
 * the answer to "given the model said this, how often is it right", and
 * that question is not being asked about a class the model stayed quiet
 * on — printing it there would be four irrelevant numbers competing with
 * the one that matters.
 * ============================================================
 */

function PredictionRow({ prediction, threshold, weak }) {
  const point = prediction.operating_point || {};
  const positive = Boolean(prediction.positive);
  const falseRate = falseCallRate(point.precision);

  return (
    <div
      className={`cv-ecg-prediction ${positive ? "positive" : ""} ${
        weak ? "weak" : ""
      }`}
    >
      <div className="cv-ecg-prediction-head">
        <div className="cv-ecg-prediction-name">
          <strong>{prediction.label || prediction.name}</strong>

          <code>{prediction.name}</code>

          {weak && <span className="cv-inline-badge warn">Weak class</span>}
        </div>

        <div className="cv-ecg-prediction-value">
          <strong>{formatProbability(prediction.probability)}</strong>

          <span className={positive ? "called" : ""}>
            {positive ? "Called positive" : "Below threshold"}
          </span>
        </div>
      </div>

      <div className="cv-ecg-prediction-bar">
        <i style={{ width: `${(prediction.probability ?? 0) * 100}%` }} />

        <b
          style={{ left: `${threshold * 100}%` }}
          title={`Calling threshold p = ${threshold}`}
        />
      </div>

      <div className="cv-ecg-prediction-detail">{prediction.description}</div>

      {positive && (
        <div className="cv-ecg-operating-point">
          <span>
            At this threshold on the held-out test split, this class scored
            precision <strong>{formatMetric(point.precision)}</strong> and
            recall <strong>{formatMetric(point.recall)}</strong> (AUROC{" "}
            {formatMetric(point.auroc)}, AP{" "}
            {formatMetric(point.average_precision)}, prevalence{" "}
            {formatMetric(point.test_prevalence)}).
            {falseRate !== null && (
              <>
                {" "}
                About <strong>{falseRate}%</strong> of positive calls for this
                class were wrong on that split.
              </>
            )}
          </span>

          {prediction.caveat && (
            <span className="cv-ecg-caveat">{prediction.caveat}</span>
          )}
        </div>
      )}
    </div>
  );
}

export default EcgResult;
