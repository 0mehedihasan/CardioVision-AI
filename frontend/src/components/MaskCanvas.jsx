import { useEffect, useMemo, useRef, useState } from "react";

/*
 * ============================================================
 * MASK CANVAS
 *
 * Renders the raw segmentation mask returned by the backend on a
 * <canvas> layered over the original frame, so classes can be toggled
 * and blended interactively.
 *
 * The mask arrives as a flat row-major Uint8 array where each entry is a
 * class index: index = y * width + x.
 * ============================================================
 */

function MaskCanvas({ mask, baseImage, classNames = {} }) {
  const canvasRef = useRef(null);

  const foregroundClasses = useMemo(() => {
    if (!mask?.data) return [];

    const present = new Set();

    for (let index = 0; index < mask.data.length; index += 1) {
      const value = mask.data[index];
      if (value > 0) present.add(value);
    }

    return [...present].sort((a, b) => a - b);
  }, [mask]);

  const [hidden, setHidden] = useState(() => new Set());
  const [opacity, setOpacity] = useState(0.5);

  // A new mask means a new analysis. Carrying the previous run's hidden
  // classes over would silently omit a structure from the canvas while the
  // legend still listed it — the viewer would be looking at an incomplete
  // segmentation with no indication that anything was withheld.
  useEffect(() => {
    setHidden(new Set());
  }, [mask]);

  const toggleClass = (classIndex) => {
    setHidden((previous) => {
      const next = new Set(previous);

      if (next.has(classIndex)) {
        next.delete(classIndex);
      } else {
        next.add(classIndex);
      }

      return next;
    });
  };

  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas || !mask?.data) return;

    const { width, height, data } = mask;

    canvas.width = width;
    canvas.height = height;

    const context = canvas.getContext("2d");
    if (!context) return;

    const imageData = context.createImageData(width, height);
    const pixels = imageData.data;

    for (let index = 0; index < data.length; index += 1) {
      const classIndex = data[index];
      const offset = index * 4;

      if (classIndex === 0 || hidden.has(classIndex)) {
        pixels[offset + 3] = 0;
        continue;
      }

      const color =
        mask.class_colors?.[String(classIndex)] ?? [255, 255, 255];

      pixels[offset] = color[0];
      pixels[offset + 1] = color[1];
      pixels[offset + 2] = color[2];
      pixels[offset + 3] = 255;
    }

    context.putImageData(imageData, 0, 0);
  }, [mask, hidden]);

  if (!mask?.data) {
    return (
      <div className="cv-mask-canvas-empty">
        The raw mask was not included in this response.
      </div>
    );
  }

  return (
    <div className="cv-mask-canvas">
      <div className="cv-mask-stage">
        {baseImage && (
          <img
            src={baseImage}
            alt="Echocardiography frame"
            className="cv-mask-base"
          />
        )}

        <canvas
          ref={canvasRef}
          className="cv-mask-layer"
          style={{ opacity }}
        />
      </div>

      <div className="cv-mask-controls">
        <div className="cv-mask-legend">
          {foregroundClasses.length === 0 && (
            <span className="cv-mask-none">
              No structures were segmented in this frame.
            </span>
          )}

          {foregroundClasses.map((classIndex) => {
            const color =
              mask.class_colors?.[String(classIndex)] ?? [255, 255, 255];

            const isHidden = hidden.has(classIndex);

            return (
              <button
                key={classIndex}
                type="button"
                className={`cv-mask-toggle ${isHidden ? "off" : ""}`}
                onClick={() => toggleClass(classIndex)}
                aria-pressed={!isHidden}
              >
                <i
                  style={{
                    background: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
                  }}
                />

                {classNames[classIndex] || `Class ${classIndex}`}
              </button>
            );
          })}
        </div>

        <label className="cv-mask-opacity">
          Overlay opacity

          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(event) =>
              setOpacity(Number(event.target.value))
            }
          />

          <span>{Math.round(opacity * 100)}%</span>
        </label>
      </div>
    </div>
  );
}

export default MaskCanvas;
