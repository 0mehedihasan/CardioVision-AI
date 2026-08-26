"""
HTTP layer.

:func:`cardiovision.api.app.create_app` builds the FastAPI application from the
routers in :mod:`cardiovision.api.routers`. Import it lazily — importing this
package pulls in FastAPI, and several test suites deliberately run without it.
"""
