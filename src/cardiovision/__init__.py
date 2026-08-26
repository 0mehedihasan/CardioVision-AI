"""
CardioVision AI — a locally deployed cardiovascular AI platform.

Two models are trained and serving: a UNet++ / EfficientNet-B3 network that
segments four cardiac structures from echocardiography, and a 1-D ResNet that
screens 12-lead ECGs for five diagnostic superclasses. MedGemma answers
case-level clinical questions locally and can be given the real model output
as context.

Nothing here fabricates a result. The CCTA, clinical-risk and multimodal-fusion
pipelines are untrained, and :data:`cardiovision.config.MODALITY_STATUS` says so
as the single source of truth the API reports from.

Layout
------
``config``          paths, device selection, model constants, real metrics
``preprocessing``   turning uploaded files into model input
``inference``       the models themselves, plus saliency
``rendering``       server-side PNG and SVG generation
``services``        auth, case storage, prompt construction
``api``             FastAPI app, routers and schemas
``cli``             ``cardiovision serve``
"""

__version__ = "4.0.0"

__all__ = ["__version__"]
