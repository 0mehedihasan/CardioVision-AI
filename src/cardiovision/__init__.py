"""
CardioVision AI — a locally deployed cardiovascular AI platform.

Three models are trained and serving: a UNet++ / EfficientNet-B3 network that
segments four cardiac structures from echocardiography, a Small3DUNet that
segments the coronary lumen from CCTA volumes, and a 1-D ResNet that screens
12-lead ECGs for five diagnostic superclasses. MedGemma answers case-level
clinical questions locally and writes the narrative section of the integrated
report; both are given the real model output as context.

Nothing here fabricates a result. The clinical-risk and multimodal-fusion
pipelines are untrained — ``fusion`` is a deterministic evidence aggregator, not
a learned model — and :data:`cardiovision.config.MODALITY_STATUS` says so as the
single source of truth the API reports from.

Layout
------
``config``          paths, device selection, model constants, real metrics
``preprocessing``   turning uploaded files into model input
``inference``       the models themselves, plus saliency
``rendering``       server-side PNG and SVG generation
``fusion``          deterministic cross-modal evidence and the report schema
``services``        auth, case storage, prompt construction
``api``             FastAPI app, routers and schemas
``cli``             ``cardiovision serve``
"""

__version__ = "4.0.0"

__all__ = ["__version__"]
