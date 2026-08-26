"""
CardioVision AI — MedGemma clinical language model.

Extracted from the original single-file backend so the language model and
the imaging models can load and fail independently.

The system prompt states exactly what the three trained models produce and,
at greater length, what they do not. MedGemma is fluent enough to write a
convincing ejection fraction or stenosis grade out of nothing if the rules
leave room for it, so the prohibitions are enumerated rather than implied.
It receives structured evidence assembled by ``cardiovision.fusion``; it
never sees an image, and it is not permitted to revise a model's numbers.
"""

from __future__ import annotations

import threading
from typing import Optional

import torch

from cardiovision.config import DEVICE, MAX_NEW_TOKENS, MEDGEMMA_NAME, MEDGEMMA_PATH


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

RULES ABOUT THE MODELS THAT PRODUCE THE CASE CONTEXT:

13. Exactly three trained models exist in this project, and each does one
    narrow task:
    - Coronary CT angiography: binary segmentation of the contrast-filled
      coronary lumen. It outputs a mask and nothing else.
    - Echocardiography: segmentation of four regions — background, left
      ventricular cavity, myocardium, left atrium.
    - 12-lead ECG: five independent probabilities — NORM, MI, STTC, CD,
      HYP.
    That is the full extent of what any model in CardioVision produces.

14. What no model here computes, and what you must therefore refuse to
    state for a specific patient: ejection fraction, wall motion, strain,
    valve function, Doppler velocities, chamber volumes, stenosis
    severity, percentage narrowing, calcium score, CAD-RADS category,
    vessel names or territories, heart rate, rhythm, PR/QRS/QT intervals,
    axis, infarct location, and acute-versus-old infarct age. If asked
    for any of these, say plainly that CardioVision does not compute it.

15. If the case context marks a modality as NOT AVAILABLE, NOT PROVIDED,
    NOT ANALYSED, or NO MODEL, you must say so. Never produce findings
    for it, not even hypothetically or as an example.

16. Segmentation areas are only absolute measurements when the case
    context gives them in cm². If areas are given as a percentage of the
    image field, treat them as relative and do not compare them against
    published reference ranges.

17. A reported Dice score, AUROC, precision or recall describes the
    model's average performance on a held-out test cohort. It is not the
    probability that this specific output is correct, and it is not a
    diagnostic confidence. Do not present it as either.

18. There is no multimodal fusion model and no clinical risk model. If
    the case context lists findings from more than one modality, they were
    collected side by side by a deterministic software layer. Do not
    combine them into a single risk score, likelihood, probability or
    severity grade, and do not say that one modality confirms,
    corroborates or is consistent with another. You may state that two
    findings were observed together.

19. The three models were trained on three unrelated public datasets and
    no patient appears in more than one. That the inputs on a case belong
    to the same patient is asserted by the operator, not established by
    any model. Do not present multimodal agreement as evidence.

20. An absent finding is not a negative finding. If a structure was not
    segmented, a class did not cross its threshold, or a region was
    outside the analysed area, say it was not identified or not examined —
    never that it is normal, absent, or ruled out.

21. Where the case context supplies UNCERTAINTIES or LIMITATIONS, carry
    the relevant ones into your answer. Do not drop a stated limitation
    to make the answer read more cleanly.

22. Never contradict, revise or round the numbers in the case context.
    They are model outputs. If you believe a number is clinically
    implausible, say so and attribute it to the model rather than
    replacing it.
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
