import { useState } from "react";

import MaskCanvas from "./MaskCanvas";

/*
 * ============================================================
 * ECHO RESULT
 *
 * Displays the real output of the trained UNet++ / EfficientNet-B3
 * echocardiography segmentation model.
 *
 * Every number rendered here comes from the backend response. Model
 * accuracy figures are labelled as dataset-level so they cannot be
 * mistaken for per-prediction confidence.
 * ============================================================
 */

const VIEWS = [
  ["overlay", "Overlay"],
  ["mask", "Segmentation"],
  ["saliency_overlay", "Saliency"],
  ["combined", "Combined"],
  ["interactive", "Interactive"],
];

const ROTATIONS = [0, 90, 180, 270];

function formatArea(structure, calibrated) {
  if (!structure.present) return "Not identified";

  if (calibrated && structure.area_cm2) {
    return `${structure.area_cm2.toFixed(2)} cm²`;
  }

  return `${structure.area_percent.toFixed(1)}% of field`;
}

function EchoResult({
  result,
  rotate = 0,
  flip = false,
  onReorient,
  isAnalyzing = false,
}) {
  const model = result.model || {};
  const metrics = model.metrics || {};
  const input = result.input || {};
  const explainability = result.explainability || {};
  const orientation = result.orientation || {};
  const quantification = result.quantification || {};

  const saliencyAvailable = Boolean(explainability.available);

  // Saliency views are only offered when a real gradient map exists. An
  // all-zero gradient still renders as a smooth colourful heatmap, which
  // would look exactly like a working explanation.
  const views = saliencyAvailable
    ? VIEWS
    : VIEWS.filter(
        ([id]) => id !== "saliency_overlay" && id !== "combined"
      );

  const [view, setView] = useState("overlay");

  // If the previous run had saliency and this one does not, the remembered
  // tab would point at a view that no longer exists.
  const activeView = views.some(([id]) => id === view) ? view : "overlay";

  const calibrated = Boolean(input.has_spatial_calibration);
  const structures = result.structures || [];

  const classNames = model.class_names || {};

  const legend = structures.map((structure) => [
    structure.name,
    result.mask?.class_colors?.[String(structure.class_index)] || [
      148, 163, 184,
    ],
  ]);

  return (
    <div className="cv-echo-result">
      {/* ====================================================
          VISUALS
      ==================================================== */}

      <div className="cv-echo-visual">
        <div className="cv-echo-viewtabs" role="tablist">
          {views.map(([id, label]) => (
            <button
              key={id}
              role="tab"
              aria-selected={activeView === id}
              className={activeView === id ? "active" : ""}
              onClick={() => setView(id)}
            >
              {label}
            </button>
          ))}
        </div>

        {activeView === "interactive" ? (
          <MaskCanvas
            mask={result.mask}
            baseImage={result.images?.original}
            classNames={classNames}
          />
        ) : (
          <div className="cv-echo-image-frame">
            <img
              src={result.images?.[activeView]}
              alt={`Echocardiography ${activeView.replace("_", " ")}`}
            />
          </div>
        )}

        <div className="cv-echo-legend">
          {legend.map(([name, rgb]) => (
            <span key={name}>
              <i style={{ background: `rgb(${rgb.join(", ")})` }} /> {name}
            </span>
          ))}
        </div>

        {activeView === "saliency_overlay" && (
          <div className="cv-echo-caption">
            <strong>{explainability.method}</strong>

            <span>{explainability.description}</span>
          </div>
        )}

        {!saliencyAvailable && (
          <div className="cv-echo-warning">
            No attribution map was produced for this run, so the saliency
            views are hidden rather than filled with an empty heatmap. The
            segmentation itself is unaffected.
          </div>
        )}
      </div>

      {/* ====================================================
          ORIENTATION
      ==================================================== */}

      <OrientationControl
        orientation={orientation}
        rotate={rotate}
        flip={flip}
        onReorient={onReorient}
        isAnalyzing={isAnalyzing}
      />

      {/* ====================================================
          DETAIL
      ==================================================== */}

      <div className="cv-echo-detail">
        <div className="cv-echo-block">
          <span className="cv-card-kicker">Segmented structures</span>

          <table className="cv-structure-table">
            <thead>
              <tr>
                <th>Structure</th>
                <th>Size</th>
                <th>Mean probability</th>
              </tr>
            </thead>

            <tbody>
              {structures.map((structure) => (
                <tr
                  key={structure.class_index}
                  className={structure.present ? "" : "absent"}
                >
                  <td>
                    <span
                      className="cv-structure-dot"
                      style={{
                        background: `rgb(${(
                          result.mask?.class_colors?.[
                            String(structure.class_index)
                          ] || [148, 163, 184]
                        ).join(", ")})`,
                      }}
                    />

                    {structure.name}
                  </td>

                  <td>{formatArea(structure, calibrated)}</td>

                  <td>
                    {structure.present
                      ? structure.mean_confidence.toFixed(3)
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {!calibrated && (
            <div className="cv-echo-warning">
              This image carried no pixel spacing, so sizes are relative
              to the image field and not absolute measurements. Upload
              NIfTI or DICOM to get areas in cm².
            </div>
          )}

          {quantification.presence_threshold_pixels && (
            <div className="cv-threshold-note">
              <strong>How “Not identified” is decided</strong>

              <span>
                A structure counts as present once at least{" "}
                {quantification.presence_threshold_pixels} pixels of the{" "}
                {quantification.mask_size}×{quantification.mask_size} mask
                carry its label; below that the region is treated as
                segmentation noise. “Not identified” therefore means the
                model did not label enough pixels — it is not a clinical
                statement that the structure is absent.
              </span>
            </div>
          )}
        </div>

        {/* ==================================================
            MODEL CARD
        ================================================== */}

        <div className="cv-echo-block">
          <span className="cv-card-kicker">Model</span>

          <div className="cv-model-grid">
            <div>
              <span>Architecture</span>
              <strong>
                {model.architecture} / {model.encoder}
              </strong>
            </div>

            <div>
              <span>Input</span>
              <strong>
                {model.input_size} × {model.input_size}, {model.num_classes}{" "}
                classes
              </strong>
            </div>

            <div>
              <span>Parameters</span>
              <strong>
                {model.parameters
                  ? model.parameters.toLocaleString()
                  : "—"}
              </strong>
            </div>

            <div>
              <span>Best epoch</span>
              <strong>{model.best_epoch ?? "—"}</strong>
            </div>
          </div>

          <div className="cv-metrics-note">
            <strong>Held-out test performance</strong>

            <span>
              Measured on {metrics.test_patients} unseen patients (
              {metrics.test_pairs} image pairs) from {metrics.dataset}, split
              at patient level. These describe the model overall — they are
              not a confidence score for this particular segmentation.
            </span>
          </div>

          <div className="cv-metric-grid">
            <div className="cv-metric">
              <span>Test Dice</span>
              <strong>{metrics.test_dice}</strong>
            </div>

            <div className="cv-metric">
              <span>Test IoU</span>
              <strong>{metrics.test_iou}</strong>
            </div>
          </div>

          {/* Validation numbers are kept visually separate. Grouping them
              under "Held-out test performance" implied the validation split
              was also held out, which it was not — it steered training
              through early stopping and checkpoint selection. */}
          <div className="cv-metrics-note secondary">
            <strong>Validation split (used during training)</strong>

            <span>
              Read back from the checkpoint itself. This split selected the
              best epoch and triggered early stopping, so it is not an
              independent estimate of performance — the test figures above
              are.
            </span>
          </div>

          <div className="cv-metric-grid">
            <div className="cv-metric muted">
              <span>Validation Dice</span>
              <strong>{metrics.validation_dice ?? "—"}</strong>
            </div>

            <div className="cv-metric muted">
              <span>Validation IoU</span>
              <strong>{metrics.validation_iou ?? "—"}</strong>
            </div>
          </div>

          {metrics.per_class_test_dice && (
            <div className="cv-perclass">
              <span className="cv-card-kicker">Per-class test Dice</span>

              <div>
                {Object.entries(metrics.per_class_test_dice).map(
                  ([name, value]) => (
                    <div key={name} className="cv-perclass-row">
                      <span>{name}</span>

                      <div className="cv-perclass-bar">
                        <i style={{ width: `${value * 100}%` }} />
                      </div>

                      <strong>{value.toFixed(4)}</strong>
                    </div>
                  )
                )}
              </div>
            </div>
          )}
        </div>

        {/* ==================================================
            INPUT PROVENANCE
        ================================================== */}

        <div className="cv-echo-block">
          <span className="cv-card-kicker">Input</span>

          <div className="cv-model-grid">
            <div>
              <span>File</span>
              <strong title={input.filename}>{input.filename}</strong>
            </div>

            <div>
              <span>Format</span>
              <strong>{(input.format || "").toUpperCase()}</strong>
            </div>

            <div>
              <span>Original size</span>
              <strong>
                {input.original_shape
                  ? `${input.original_shape[0]} × ${input.original_shape[1]}`
                  : "—"}
              </strong>
            </div>

            <div>
              <span>Pixel spacing</span>
              <strong>
                {input.pixel_spacing_mm
                  ? `${input.pixel_spacing_mm[0].toFixed(3)} × ${input.pixel_spacing_mm[1].toFixed(3)} mm`
                  : "Not available"}
              </strong>
            </div>

            {input.frame_count && (
              <div>
                <span>Frame</span>
                <strong>
                  {input.frame_index} of {input.frame_count}
                </strong>
              </div>
            )}

            <div>
              <span>Orientation</span>
              <strong>
                {orientation.reoriented
                  ? [
                      orientation.rotation_applied
                        ? `Rotated ${orientation.rotation_applied}° CCW`
                        : null,
                      orientation.flip_applied ? "mirrored" : null,
                    ]
                      .filter(Boolean)
                      .join(", ")
                  : "As uploaded"}
              </strong>
            </div>

            <div>
              <span>Inference</span>
              <strong>
                {result.inference_ms} ms on {result.device}
                {result.configured_device &&
                  result.device !== result.configured_device && (
                    <em>
                      {" "}
                      (fell back from {result.configured_device})
                    </em>
                  )}
              </strong>
            </div>
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
          Anatomical segmentation only. This model does not measure
          ejection fraction, wall motion, valve function or haemodynamics,
          and it does not produce a diagnosis.
        </div>
      </div>
    </div>
  );
}

/*
 * ============================================================
 * ORIENTATION CONTROL
 *
 * The model learned on CAMUS arrays whose sector apex points left. A
 * screenshot or DICOM frame is normally apex-up, i.e. a quarter turn
 * away, which puts it outside the training distribution.
 *
 * The correction is offered, never applied silently: the right rotation
 * depends on how the image was exported, and a hidden transform would
 * leave no way to tell a real segmentation from a lucky one.
 * ============================================================
 */

function OrientationControl({
  orientation,
  rotate,
  flip,
  onReorient,
  isAnalyzing,
}) {
  if (!onReorient) return null;

  const needsAttention =
    orientation.display_oriented_format && !orientation.reoriented;

  return (
    <div
      className={`cv-orientation ${needsAttention ? "attention" : ""}`}
    >
      <div className="cv-orientation-head">
        <span className="cv-card-kicker">Image orientation</span>

        {needsAttention && (
          <span className="cv-orientation-flag">Check this</span>
        )}
      </div>

      <p>
        Trained on images with the{" "}
        <strong>{orientation.training_orientation}</strong>. Conventional
        echo displays are apex-up, so an exported PNG, JPEG or DICOM frame
        usually needs a quarter turn before it matches what the model
        learned. If the segmentation below looks anatomically wrong, this
        is the first thing to change.
      </p>

      <div className="cv-orientation-actions">
        <div
          className="cv-orientation-rotations"
          role="group"
          aria-label="Rotation"
        >
          {ROTATIONS.map((degrees) => (
            <button
              key={degrees}
              type="button"
              disabled={isAnalyzing}
              aria-pressed={rotate === degrees}
              className={rotate === degrees ? "active" : ""}
              onClick={() => onReorient(degrees, flip)}
            >
              {degrees === 0 ? "As uploaded" : `${degrees}°`}
            </button>
          ))}
        </div>

        <button
          type="button"
          className={`cv-orientation-flip ${flip ? "active" : ""}`}
          disabled={isAnalyzing}
          aria-pressed={flip}
          onClick={() => onReorient(rotate, !flip)}
        >
          Mirror horizontally
        </button>
      </div>

      <span className="cv-orientation-hint">
        Each change re-runs the model on the original file — nothing is
        cropped, resampled twice or altered beyond the rotation shown.
      </span>
    </div>
  );
}

export default EchoResult;
