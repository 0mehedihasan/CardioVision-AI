"""
One router per concern.

``health``  public service and model state, plus the model cards
``auth``    login, logout, session check
``echo``    echocardiography segmentation
``ecg``     12-lead ECG classification
``cases``   patient case CRUD and stored images
``qa``      MedGemma clinical questions

Route order matters in exactly one place: ``/api/cases/{case_id}`` must be
registered after any literal sibling path, or the literal gets swallowed as an
ID. There are none today; keep it that way.
"""
