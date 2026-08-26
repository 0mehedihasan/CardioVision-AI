import { useState } from "react";

/*
 * ============================================================
 * CCTA RESULT
 *
 * Displays the real output of the trained Small3DUNet coronary
 * lumen segmentation model.
 *
 * This is the weakest of the three models in the project, and this
 * component is written to say so rather than to look confident.
 * Three things drive its layout:
 *
 *   1. Coverage is reported before findings. A bounded window
 *      budget means part of the volume may never have been
 *      examined, and zero there means "not looked at" — never
 *      "nothing found". That distinction goes above the numbers,
 *      not in a footnote below them.
 *
 *   2. A slice is not the result. Every panel is labelled with
 *      the plane and index it came from, and the projection view
 *      is offered alongside because it is the only view where a
 *      fragmented mask is visible.
 *
 *   3. The test split was three cases. The Dice score is shown
 *      with that denominator attached, so it cannot be read as
 *      per-case confidence.
 * ============================================================
 */

/* The loader does not reorient the volume, so these are array axes and are
   labelled as such. Calling axis 2 "axial" would be a claim about patient
   orientation that nothing in this pipeline verifies. */
const PLANES = [
  ["axis0", "Plane 0"],
  ["axis1", "Plane 1"],
  ["axis2", "Plane 2"],
];

const LAYERS = [
  ["overlay", "CT + lumen"],
  ["ct", "CT only"],
  ["probability", "Probability"],
  ["mip", "Projection"],
];

const LAYER_CAPTIONS = {
  overlay: "Predicted lumen overlaid on the CT slice.",
  ct: "The CT slice as the model saw it, windowed for display only.",
  probability:
    "Per-voxel sigmoid output before thresholding. Warm is high probability.",
  mip:
    "Maximum-intensity projection of the probability map through the whole " +
    "volume. This is the view where a fragmented prediction shows up.",
};

/* The lumen colour is fixed in config (amber) and echoed here for the legend
   so the swatch cannot drift from the render. */
const LUMEN_COLOR = "rgb(250, 204, 21)";

function formatProbability(value) {
  return typeof value === "number" ? value.toFixed(3) : "—";
}

function formatMetric(value, digits = 3) {
  return typeof value === "number" ? value.toFixed(digits) : "—";
}

function formatPercent(value, digits = 2) {
  return typeof value === "number" ? `${value.toFixed(digits)}%` : "—";
}

function formatCount(value) {
  return typeof value === "number" ? value.toLocaleString() : "—";
}

function formatVolume(value) {
  if (typeof value !== "number") return "—";

  // Sub-millilitre volumes are the normal case for a coronary lumen, and
  // 0.00 mL would read as nothing found.
  if (value > 0 && value < 0.01) return "< 0.01 mL";

  return `${value.toFixed(2)} mL`;
}

function formatShape(shape) {
  return Array.isArray(shape) ? shape.join(" × ") : "—";
}

function formatSpacing(spacing) {
  return Array.isArray(spacing)
    ? `${spacing.map((value) => Number(value).toFixed(2)).join(" × ")} mm`
    : "—";
}

function CctaResult({ result }) {
  const model = result.model || {};
  const metrics = model.metrics || {};
  const threshold = model.threshold || {};
  const modelInput = model.input || {};

  const input = result.input || {};
  const coverage = result.coverage || {};
  const quantification = result.quantification || {};
  const explainability = result.explainability || {};
  const figures = result.figures || {};
  const figureMeta = result.figure_meta || {};
  const planeMeta = figureMeta.planes || {};

  const findings = result.findings || [];
  const lumen = findings[0] || null;

  const complete = coverage.complete !== false;
  const analysedPercent = coverage.analysed_percent;

  const gradcamAvailable = Boolean(explainability.available) &&
    Boolean(figures.gradcam_overlay);

  /* Only planes that actually rendered are offered. A tab pointing at a
     missing figure would show a broken frame where the honest answer is that
     this panel could not be produced. */
  const planes = PLANES.filter(
    ([id]) => figures[`${id}_overlay`] || figures[`${id}_mip`]
  );

  const [plane, setPlane] = useState(planes[0]?.[0] || "axis0");
  const [layer, setLayer] = useState("overlay");

  const activePlane = planes.some(([id]) => id === plane)
    ? plane
    : planes[0]?.[0];

  const layers = LAYERS.filter(
    ([id]) => figures[`${activePlane}_${id}`]
  );

  const activeLayer = layers.some(([id]) => id === layer)
    ? layer
    : layers[0]?.[0];

  const activeFigure = figures[`${activePlane}_${activeLayer}`];
  const activeMeta = planeMeta[activePlane] || {};

  /* Nothing was inferred, so there is nothing to show. This happens when a
     JSON-only response is rendered, or when the analysis returned but every
     panel failed. */
  if (!result.analyzed) {
    return (
      <div className="cv-empty-analysis">
        <p>No coronary CT volume was analysed in this case.</p>
      </div>
    );
  }

  const fragmented =
    typeof lumen?.components === "number" && lumen.components > 12;

  return (
    <div className="cv-ccta-result">
      {/* ====================================================
          HEADLINE
      ==================================================== */}

      <div className="cv-ecg-headline">
        <div>
          <span className="cv-card-kicker">Coronary lumen segmentation</span>

          <h3>
            {lumen?.present
              ? `Coronary lumen segmented — ${formatVolume(lumen.volume_ml)} ` +
                `across ${formatCount(lumen.voxels)} voxels`
              : "No coronary lumen was segmented above the presence threshold"}
          </h3>

          <p>
            The model produces a <strong>binary lumen mask</strong> and nothing
            else. There is no stenosis grade, no calcium score and no vessel
            identity in this output.{" "}
            {complete
              ? "The whole volume was covered by the sliding window."
              : `Only ${formatPercent(analysedPercent)} of the volume was ` +
                "examined — everything outside that region is unknown, not " +
                "negative."}
          </p>
        </div>

        <div className="cv-ecg-calls">
          <div className={`cv-ecg-call ${complete ? "" : "weak"}`}>
            {complete ? "Full coverage" : "Partial coverage"}

            <em>
              {formatCount(coverage.windows_run)} of{" "}
              {formatCount(coverage.windows_total)} windows
            </em>
          </div>

          <div className={`cv-ecg-call ${lumen?.present ? "" : "none"}`}>
            {lumen?.present ? "Lumen present" : "Below threshold"}

            <em>at p &gt; {formatMetric(result.threshold, 2)}</em>
          </div>
        </div>
      </div>

      {/* ====================================================
          COVERAGE

          Above the findings on purpose. A number measured over
          38% of a volume means something different from the same
          number measured over all of it, and the reader has to
          know which before reading it.
      ==================================================== */}

      {!complete ? (
        <div className="cv-ecg-warning">
          <strong>
            {formatPercent(analysedPercent)} of this volume was analysed
          </strong>

          <span>
            {coverage.note ||
              "Everything outside the analysed region was not examined."}{" "}
            The window budget covered {formatCount(coverage.windows_run)} of{" "}
            {formatCount(coverage.windows_total)} positions at{" "}
            {formatShape(coverage.patch_size)} with{" "}
            {formatPercent((coverage.overlap ?? 0) * 100, 0)} overlap, so a
            centred crop was analysed and the rest was left alone. Raise the
            window budget, or crop the study to the cardiac region, to cover
            it fully.
          </span>
        </div>
      ) : (
        <div className="cv-ecg-note">
          <strong>Coverage complete.</strong> {coverage.note} All{" "}
          {formatCount(coverage.windows_total)} sliding-window positions were
          run, so a zero in the mask is a prediction of no lumen rather than
          an unexamined region.
        </div>
      )}

      {/* ====================================================
          SLICE VIEWS
      ==================================================== */}

      {activeFigure ? (
        <div className="cv-echo-visual">
          <div className="cv-echo-viewtabs" role="tablist">
            {planes.map(([id, label]) => (
              <button
                key={id}
                role="tab"
                aria-selected={activePlane === id}
                className={activePlane === id ? "active" : ""}
                onClick={() => setPlane(id)}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="cv-echo-viewtabs" role="tablist">
            {layers.map(([id, label]) => (
              <button
                key={id}
                role="tab"
                aria-selected={activeLayer === id}
                className={activeLayer === id ? "active" : ""}
                onClick={() => setLayer(id)}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="cv-echo-image-frame">
            <img
              src={activeFigure}
              alt={`Coronary CT ${activePlane} ${activeLayer}`}
            />
          </div>

          <div className="cv-echo-legend">
            <span>
              <i style={{ background: LUMEN_COLOR }} />
              Coronary artery lumen
            </span>

            <span>
              <i style={{ background: "#0b0f12", border: "1px solid #334155" }} />
              Background or not analysed
            </span>
          </div>

          <div className="cv-echo-caption">
            <strong>
              {activeLayer === "mip"
                ? `${PLANES.find(([id]) => id === activePlane)?.[1]} projection ` +
                  "— whole volume"
                : `${PLANES.find(([id]) => id === activePlane)?.[1]} slice ` +
                  `${formatCount(activeMeta.index)} of ` +
                  `${formatCount(activeMeta.of)}`}
            </strong>

            <span>{LAYER_CAPTIONS[activeLayer]}</span>

            {activeLayer !== "mip" && activeMeta.selected_by && (
              <span>
                Slice chosen by {activeMeta.selected_by};{" "}
                {formatCount(activeMeta.lumen_voxels_in_slice)} lumen voxels in
                this plane. {figureMeta.display_note}
              </span>
            )}

            {activeLayer === "mip" && figureMeta.projection_note && (
              <span>{figureMeta.projection_note}</span>
            )}

            {figureMeta.axis_note && <span>{figureMeta.axis_note}</span>}
          </div>

          {/* Per-slice coverage, not just per-volume. A slice can sit almost
              entirely outside the analysed crop even when the volume-level
              figure looks reasonable. */}
          {activeLayer !== "mip" && activeMeta.warning && (
            <div className="cv-ecg-note warn">
              <strong>Partly outside the analysed region.</strong>{" "}
              {activeMeta.warning}
            </div>
          )}
        </div>
      ) : (
        <div className="cv-xai-unavailable">
          <strong>No slice views were rendered</strong>

          <p>
            The measurements below come from the mask and are unaffected, but
            there is no picture to check them against in this response.
          </p>
        </div>
      )}

      {/* ====================================================
          MEASUREMENTS
      ==================================================== */}

      <div className="cv-ecg-block">
        <span className="cv-card-kicker">Measurements</span>

        <p className="cv-ecg-block-intro">
          {lumen?.present ? (
            <>
              Everything measurable about the predicted mask. These are{" "}
              <strong>geometry, not physiology</strong> — they describe how much
              voxel the model marked as lumen at the operating point below, and
              nothing about flow, stenosis or vessel identity.
            </>
          ) : (
            <>
              The mask fell below the presence threshold. That is a{" "}
              <strong>size cutoff on the model&rsquo;s output</strong>, not a
              clinical finding about this patient.
            </>
          )}
        </p>

        {lumen ? (
          <>
            <div className="cv-metric-grid">
              <div className="cv-metric">
                <span>Lumen volume</span>

                <strong>{formatVolume(lumen.volume_ml)}</strong>
              </div>

              <div className="cv-metric">
                <span>Voxels</span>

                <strong>{formatCount(lumen.voxels)}</strong>
              </div>

              <div className="cv-metric">
                <span>Of analysed region</span>

                <strong>{formatPercent(lumen.percent_of_analysed, 3)}</strong>
              </div>

              <div className="cv-metric">
                <span>Mean probability</span>

                <strong>{formatProbability(lumen.mean_probability)}</strong>
              </div>

              <div className="cv-metric">
                <span>Peak probability</span>

                <strong>{formatProbability(lumen.max_probability)}</strong>
              </div>

              <div className={`cv-metric ${fragmented ? "warning" : ""}`}>
                <span>Connected components</span>

                <strong>{formatCount(lumen.components)}</strong>
              </div>

              <div className="cv-metric">
                <span>Largest component</span>

                <strong>
                  {typeof lumen.largest_component_fraction === "number"
                    ? formatPercent(
                        lumen.largest_component_fraction * 100,
                        1
                      )
                    : "—"}
                </strong>
              </div>

              <div className="cv-metric">
                <span>Threshold</span>

                <strong>{formatMetric(result.threshold, 2)}</strong>
              </div>
            </div>

            {/* A coronary tree is a few connected vessels. Dozens of
                components is the model's characteristic failure, and it is
                visible in the numbers before it is visible in a slice. */}
            {fragmented && (
              <div className="cv-ecg-note warn">
                <strong>The prediction is fragmented.</strong> The mask splits
                into {formatCount(lumen.components)} disconnected components,
                with the largest holding{" "}
                {formatPercent(
                  (lumen.largest_component_fraction ?? 0) * 100,
                  1
                )}{" "}
                of the volume. A real coronary tree is a small number of
                connected vessels, so treat this output as unreliable and
                check the projection view above.
              </div>
            )}

            <div className="cv-metrics-note">
              <strong>How these numbers were computed</strong>

              <span>{quantification.note}</span>
            </div>

            <div className="cv-metrics-note secondary">
              <strong>Presence threshold</strong>

              <span>
                A mask smaller than{" "}
                {formatCount(quantification.presence_threshold_voxels)} voxels
                — about half a millilitre on this grid — is reported as not
                present. That cutoff exists because a handful of scattered
                false positives is not a coronary tree, and it is a decision
                about the model&rsquo;s output rather than about the patient.
              </span>
            </div>
          </>
        ) : (
          <p className="cv-muted-text">
            The response carried no findings, so there is nothing to measure.
          </p>
        )}
      </div>

      {/* ====================================================
          EXPLAINABILITY

          One patch, not the volume. Showing a CAM without that
          scope attached invites reading it as a localisation of
          disease across the whole study.
      ==================================================== */}

      <div className="cv-ecg-block">
        <span className="cv-card-kicker">Explainability</span>

        {gradcamAvailable ? (
          <>
            <p className="cv-ecg-block-intro">
              {explainability.method} computed over{" "}
              <strong>one 96³ patch</strong> — the patch containing the most
              predicted lumen. {explainability.scope}
            </p>

            <div className="cv-xai-grid">
              <div className="cv-xai-image">
                <div className="cv-echo-image-frame">
                  <img
                    src={figures.gradcam_overlay}
                    alt="3-D Grad-CAM overlaid on the CT patch"
                  />
                </div>

                <div className="cv-image-caption">
                  <strong>Attention over CT</strong>

                  <span>
                    {figureMeta.gradcam
                      ? `Patch slice ${formatCount(
                          figureMeta.gradcam.slice_index_in_patch
                        )}, volume slice ${formatCount(
                          figureMeta.gradcam.slice_index_in_volume
                        )}`
                      : "Slice with the most attention in the patch"}
                  </span>
                </div>
              </div>

              {figures.gradcam_mask_overlay && (
                <div className="cv-xai-image">
                  <div className="cv-echo-image-frame">
                    <img
                      src={figures.gradcam_mask_overlay}
                      alt="Predicted lumen in the same patch"
                    />
                  </div>

                  <div className="cv-image-caption">
                    <strong>Predicted lumen, same slice</strong>

                    <span>
                      What the model output there, for comparison with what it
                      attended to
                    </span>
                  </div>
                </div>
              )}
            </div>

            <div className="cv-metric-grid">
              <div className="cv-metric">
                <span>Target layer</span>

                <strong>{explainability.target_layer || "—"}</strong>
              </div>

              <div className="cv-metric">
                <span>Patch origin</span>

                <strong>{formatShape(explainability.origin)}</strong>
              </div>

              <div className="cv-metric">
                <span>Patch shape</span>

                <strong>{formatShape(explainability.shape)}</strong>
              </div>

              <div className="cv-metric">
                <span>Computed on</span>

                <strong>
                  {(explainability.device || "—").toUpperCase()}
                </strong>
              </div>
            </div>

            <div className="cv-ecg-note">
              <strong>What this is not.</strong> {explainability.note}{" "}
              Attention outside this one patch was never computed, so a dark
              region elsewhere in the study says nothing at all.
            </div>
          </>
        ) : (
          /* A zero CAM renders as a smooth, plausible heatmap. Showing it
             would be an explanation of a computation that never happened. */
          <div className="cv-xai-unavailable">
            <strong>Attribution unavailable for this run</strong>

            <p>
              No Grad-CAM was produced, so there is nothing to show. The
              segmentation and the measurements above are unaffected — only
              the explanation is missing.
            </p>
          </div>
        )}
      </div>

      {/* ====================================================
          MODEL
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
            <span>Parameters</span>

            <strong>{formatCount(model.parameters)}</strong>
          </div>

          <div>
            <span>Best epoch</span>

            <strong>{model.best_epoch ?? "—"}</strong>
          </div>

          <div>
            <span>Patch size</span>

            <strong>{formatShape(modelInput.patch_size)}</strong>
          </div>

          <div>
            <span>Target spacing</span>

            <strong>{formatSpacing(modelInput.target_spacing_mm)}</strong>
          </div>

          <div>
            <span>Inference device</span>

            <strong>{(result.device || "—").toUpperCase()}</strong>
          </div>

          <div>
            <span>Inference time</span>

            <strong>
              {typeof result.inference_ms === "number"
                ? `${(result.inference_ms / 1000).toFixed(1)} s`
                : "—"}
            </strong>
          </div>
        </div>

        {/* The threshold is a decision boundary that was chosen, not a
            property of the patient. Where it came from matters. */}
        <div className="cv-metrics-note">
          <strong>
            Operating point {formatMetric(threshold.value, 2)}
            {threshold.from_checkpoint ? " (from the checkpoint)" : ""}
          </strong>

          <span>{threshold.note}</span>
        </div>

        {/* Mean, spread and range in the same row. The backend reports all
            four because with three cases the range is the finding: a Dice of
            0.60 that runs from 0.49 to 0.73 is a different claim from 0.60
            measured tightly, and a mean alone hides that. */}
        <table className="cv-ecg-metric-table">
          <thead>
            <tr>
              <th>Held-out metric</th>

              <th>Mean</th>

              <th>SD</th>

              <th>Range over 3 cases</th>
            </tr>
          </thead>

          <tbody>
            {[
              ["Dice", metrics.test_dice, 4, false],
              ["IoU", metrics.test_iou, 4, false],
              ["Sensitivity", metrics.test_sensitivity, 4, false],
              ["Precision", metrics.test_precision, 4, false],
              /* Tinted because it is the row that the Dice score hides: a
                 95th-percentile surface distance in the tens of centimetres
                 means some predicted components are nowhere near the
                 annotation, which a respectable-looking overlap does not
                 reveal. */
              ["Hausdorff 95 (mm)", metrics.test_hd95_mm, 1, true],
            ].map(([label, spread, digits, weak]) => (
              <tr key={label} className={weak ? "weak" : ""}>
                <td>{label}</td>

                <td>{formatMetric(spread?.mean, digits)}</td>

                <td>{formatMetric(spread?.sd, digits)}</td>

                <td>
                  {typeof spread?.min === "number" &&
                  typeof spread?.max === "number"
                    ? `${formatMetric(spread.min, digits)} – ` +
                      `${formatMetric(spread.max, digits)}`
                    : "—"}
                </td>
              </tr>
            ))}

            <tr>
              <td>Validation Dice</td>

              <td>{formatMetric(metrics.validation_dice, 4)}</td>

              <td>—</td>

              <td>best epoch, 3 validation cases</td>
            </tr>
          </tbody>
        </table>

        {/* The denominator travels with the number. Three cases is not a
            test set that supports a confidence claim. */}
        <div className="cv-ecg-note warn">
          <strong>Read these as provisional.</strong> {metrics.scope}.{" "}
          {metrics.caveat}
        </div>

        <div className="cv-metrics-note secondary">
          <strong>Dataset</strong>

          <span>
            {metrics.dataset}
            {typeof metrics.dataset_cases === "number"
              ? ` — ${metrics.dataset_cases} volumes`
              : ""}
            {metrics.split ? `, split ${metrics.split}` : ""}. Nothing here was
            validated against an external cohort.
          </span>
        </div>
      </div>

      {/* ====================================================
          INPUT
      ==================================================== */}

      <div className="cv-ecg-block">
        <span className="cv-card-kicker">Input volume</span>

        <p className="cv-ecg-block-intro">
          <strong>{input.filename || "Uploaded volume"}</strong>{" "}
          {input.resampled
            ? "was resampled onto the grid the model was trained on. Every " +
              "measurement above is in that grid, not the source grid."
            : "was already on the target grid, so no resampling was needed."}
        </p>

        <div className="cv-model-grid">
          <div>
            <span>Format</span>

            <strong>{input.format || "—"}</strong>
          </div>

          <div>
            <span>Source shape</span>

            <strong>{formatShape(input.original_shape)}</strong>
          </div>

          <div>
            <span>Source spacing</span>

            <strong>{formatSpacing(input.original_spacing_mm)}</strong>
          </div>

          <div>
            <span>Analysed shape</span>

            <strong>{formatShape(input.analysed_shape)}</strong>
          </div>

          <div>
            <span>Analysed spacing</span>

            <strong>{formatSpacing(input.analysed_spacing_mm)}</strong>
          </div>

          <div>
            <span>Source slices</span>

            <strong>{formatCount(input.source_slices)}</strong>
          </div>

          <div>
            <span>HU observed</span>

            <strong>{formatShape(input.hu_range_observed)}</strong>
          </div>

          <div>
            <span>HU window used</span>

            <strong>{formatShape(input.hu_window)}</strong>
          </div>
        </div>

        {modelInput.normalization && (
          <div className="cv-metrics-note secondary">
            <strong>Normalisation</strong>

            <span>{modelInput.normalization}</span>
          </div>
        )}
      </div>

      {/* ====================================================
          LIMITATIONS AND NOTES
      ==================================================== */}

      {(result.limitations?.length > 0 || result.notes?.length > 0) && (
        <div className="cv-ecg-block">
          <span className="cv-card-kicker">Limitations and run notes</span>

          {result.limitations?.length > 0 && (
            <ul className="cv-echo-notes">
              {result.limitations.map((limitation, index) => (
                <li key={`limitation-${index}`}>{limitation}</li>
              ))}
            </ul>
          )}

          {result.notes?.length > 0 && (
            <>
              <p className="cv-ecg-block-intro">
                <strong>This run.</strong> What the loader, the model and the
                renderer reported while producing the result above.
              </p>

              <ul className="cv-echo-notes">
                {result.notes.map((note, index) => (
                  <li key={`note-${index}`}>{note}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}

      {/* ====================================================
          DISCLAIMER
      ==================================================== */}

      <div className="cv-echo-disclaimer">
        This model outputs a coronary lumen mask. It does not grade stenosis,
        score calcium, name vessels or assign a CAD-RADS category, and it was
        never validated for any of those. It is a research prototype trained on
        14 volumes and tested on 3 — not a diagnostic device, and not a
        substitute for a reader.
      </div>
    </div>
  );
}

export default CctaResult;
