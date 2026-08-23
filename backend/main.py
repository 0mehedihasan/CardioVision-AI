from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoModelForImageTextToText


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "medgemma-1.5-4b-it"

MODEL_NAME = "MedGemma 1.5 4B IT"

# 32 is too short for useful clinical answers.
MAX_NEW_TOKENS = 256


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="CardioVision AI API",
    description="Backend API for CardioVision AI clinical intelligence.",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class ClinicalQuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
    )

    context: Optional[str] = Field(
        default=None,
        max_length=12000,
    )


class ClinicalQuestionResponse(BaseModel):
    success: bool
    answer: str
    model: str
    device: str


# ============================================================
# GLOBAL MODEL OBJECTS
# ============================================================

processor = None
model = None


# ============================================================
# MODEL LOADING
# ============================================================

def load_medgemma():
    global processor
    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MedGemma model not found at: {MODEL_PATH}"
        )

    print("=" * 60)
    print("CardioVision AI")
    print("Loading MedGemma...")
    print("=" * 60)

    print(f"Model path: {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH)
    )

    print("Loading model...")

    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.bfloat16,
        device_map=DEVICE,
    )

    model.eval()

    print("MedGemma loaded successfully.")
    print(f"Model device: {next(model.parameters()).device}")
    print("=" * 60)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    load_medgemma()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "CardioVision AI",
        "medgemma_loaded": model is not None,
        "model": MODEL_NAME,
        "device": DEVICE,
    }


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(
    question: str,
    context: Optional[str] = None,
) -> str:

    system_instruction = """
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
""".strip()

    if context and context.strip():

        return f"""
{system_instruction}

CASE CONTEXT:
{context.strip()}

CLINICAL QUESTION:
{question.strip()}

Answer the question using the case context where relevant.
If the question is partly general medical knowledge, explain that
knowledge directly and then relate it to the supplied case when possible.

ANSWER:
""".strip()

    return f"""
{system_instruction}

GENERAL MEDICAL QUESTION:
{question.strip()}

Answer directly as a general medical knowledge question.

ANSWER:
""".strip()


# ============================================================
# MEDGEMMA INFERENCE
# ============================================================

def generate_answer(
    question: str,
    context: Optional[str] = None,
) -> str:

    if model is None or processor is None:
        raise RuntimeError(
            "MedGemma is not loaded."
        )

    prompt = build_prompt(
        question=question,
        context=context,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]

    formatted_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=formatted_prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[-1]

    print(
        f"Generating response for question: {question}"
    )

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    generated_tokens = output[:, input_length:]

    response = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0]

    response = response.strip()

    if not response:
        raise RuntimeError(
            "MedGemma returned an empty response."
        )

    return response


# ============================================================
# CLINICAL QUESTION ENDPOINT
# ============================================================

@app.post(
    "/api/clinical-question",
    response_model=ClinicalQuestionResponse,
)
def ask_clinical_question(
    request: ClinicalQuestionRequest,
):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Clinical question cannot be empty.",
        )

    try:

        answer = generate_answer(
            question=question,
            context=request.context,
        )

        return ClinicalQuestionResponse(
            success=True,
            answer=answer,
            model=MODEL_NAME,
            device=DEVICE,
        )

    except Exception as error:

        print(
            "=" * 60
        )
        print("MedGemma inference error")
        print(error)
        print("=" * 60)

        raise HTTPException(
            status_code=500,
            detail="Unable to generate a clinical response.",
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "service": "CardioVision AI",
        "status": "running",
        "model": MODEL_NAME,
        "device": DEVICE,
    }