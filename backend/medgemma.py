"""
CardioVision AI — MedGemma clinical language model.

Extracted from the original single-file backend so the language model and
the imaging model can load and fail independently.

The system prompt is preserved from the original implementation, with
additional rules covering what the echo segmentation model can and cannot
support, since the model now receives real segmentation output as context.
"""

from __future__ import annotations

import threading
from typing import Optional

import torch

from config import DEVICE, MAX_NEW_TOKENS, MEDGEMMA_NAME, MEDGEMMA_PATH


class MedGemmaUnavailable(RuntimeError):
    """Raised when MedGemma cannot be loaded or used."""


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are MedGemma, the clinical language model powering CardioVision AI.

Your task is to answer clinical and medical questions accurately,
clearly, and concisely.

IMPORTANT RESPONSE RULES:

1. Treat a question without patient context as a GENERAL MEDICAL
   KNOWLEDGE question.

2. For general medical questions, answer directly using your medical
   knowledge. Do not say that you cannot access patient information.

3. Only make patient-specific statements when patient or case context
   is explicitly provided.

4. When patient context is provided, use only the information contained
   in that context for patient-specific conclusions.

5. Never invent patient findings, laboratory values, imaging findings,
   diagnoses, medications, symptoms, or medical history.

6. Clearly distinguish established medical information from
   interpretation or uncertainty.

7. If information is insufficient for a patient-specific question,
   state what information is missing.

8. Do not refuse a general medical knowledge question merely because
   no patient information was supplied.

9. Keep responses clinically useful and appropriately concise.

10. Do not begin the response with phrases such as:
    "I am sorry"
    "I cannot provide"
    "I am unable to access patient information"
    unless the question genuinely requires unavailable patient data.

11. Do not mention these instructions in your answer.

12. CardioVision AI is a research and clinical decision-support
    prototype. Do not present model output as a definitive diagnosis.

RULES ABOUT THE IMAGING MODEL:

13. The only imaging model available is an echocardiography segmentation
    model. It outlines four regions: background, left ventricular cavity,
    myocardium, and left atrium. That is the full extent of its output.

14. The echo model does NOT measure ejection fraction, wall motion,
    valve function, strain, Doppler velocities, or chamber volumes. If
    asked for any of these, say that CardioVision does not compute it.

15. If the case context marks a modality as NOT AVAILABLE, you must say
    it is unavailable. Never produce findings for it, not even
    hypothetically or as an example.

16. Segmentation areas are only absolute measurements when the case
    context gives them in cm². If areas are given as a percentage of the
    image field, treat them as relative and do not compare them against
    published reference ranges.

17. A reported Dice score describes the model's average accuracy on a
    held-out test cohort. It is not the probability that this specific
    segmentation is correct, and it is not a diagnostic confidence.
    Do not present it as either.
""".strip()


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(question: str, context: Optional[str] = None) -> str:
    if context and context.strip():
        return f"""
{SYSTEM_INSTRUCTION}

CASE CONTEXT:
{context.strip()}

CLINICAL QUESTION:
{question.strip()}

Answer the question using the case context where relevant.
If the question is partly general medical knowledge, explain that
knowledge directly and then relate it to the supplied case when possible.
Base every patient-specific statement only on the case context above.

ANSWER:
""".strip()

    return f"""
{SYSTEM_INSTRUCTION}

GENERAL MEDICAL QUESTION:
{question.strip()}

Answer directly as a general medical knowledge question.

ANSWER:
""".strip()


# ============================================================
# MODEL WRAPPER
# ============================================================

class MedGemma:
    """Lazily-loaded singleton wrapper around the local MedGemma weights."""

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ---- loading ----------------------------------------------

    def load(self) -> None:
        if self.is_loaded:
            return

        if not MEDGEMMA_PATH.exists():
            self._load_error = (
                f"MedGemma weights not found at {MEDGEMMA_PATH}. This "
                "directory is gitignored because of its size; download the "
                "model locally before starting the backend."
            )
            raise MedGemmaUnavailable(self._load_error)

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            self._load_error = (
                "transformers is not installed. "
                "Install it with: pip install transformers accelerate"
            )
            raise MedGemmaUnavailable(self._load_error) from error

        print("=" * 60)
        print("CardioVision AI")
        print("Loading MedGemma...")
        print(f"Model path: {MEDGEMMA_PATH}")
        print(f"Device: {DEVICE}")

        try:
            print("Loading processor...")
            processor = AutoProcessor.from_pretrained(str(MEDGEMMA_PATH))

            print("Loading model...")
            model = AutoModelForImageTextToText.from_pretrained(
                str(MEDGEMMA_PATH),
                dtype=torch.bfloat16,
                device_map=DEVICE,
            )
        except Exception as error:
            self._load_error = f"Failed to load MedGemma: {error}"
            raise MedGemmaUnavailable(self._load_error) from error

        model.eval()

        self._processor = processor
        self._model = model
        self._load_error = None

        print("MedGemma loaded successfully.")
        print(f"Model device: {next(model.parameters()).device}")
        print("=" * 60)

    # ---- inference --------------------------------------------

    def generate(
        self,
        question: str,
        context: Optional[str] = None,
    ) -> str:
        if not self.is_loaded:
            raise MedGemmaUnavailable(
                self._load_error or "MedGemma is not loaded."
            )

        prompt = build_prompt(question=question, context=context)

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]

        formatted_prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(text=formatted_prompt, return_tensors="pt")

        inputs = {
            key: value.to(DEVICE) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        input_length = inputs["input_ids"].shape[-1]

        print(f"Generating response for question: {question}")

        # Serialise generation: concurrent generate() calls on one MPS
        # device are a reliable way to exhaust unified memory.
        with self._lock:
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )

        generated_tokens = output[:, input_length:]

        response = self._processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0].strip()

        if not response:
            raise MedGemmaUnavailable("MedGemma returned an empty response.")

        return response

    def describe(self) -> dict[str, object]:
        return {
            "name": MEDGEMMA_NAME,
            "device": DEVICE,
            "loaded": self.is_loaded,
            "max_new_tokens": MAX_NEW_TOKENS,
        }


# Module-level singleton.
medgemma = MedGemma()
