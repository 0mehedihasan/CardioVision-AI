"""
Application services: things that are neither a model nor an HTTP concern.

``auth``          fixed-user credential check and in-memory session tokens
``database``      SQLite case store plus the files that go with each case
``case_context``  rendering a case into prompt text for MedGemma

None of these import from :mod:`cardiovision.api`, so they stay testable
without FastAPI installed — which is how the case-lifecycle suite runs.
"""
