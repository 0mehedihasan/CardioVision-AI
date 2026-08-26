"""
Server-side PNG rendering.

Figures are produced on the backend, base64-encoded and handed to the browser
as data URLs, so the client never needs a plotting library and the rendered
output can be archived with the case exactly as the operator saw it.
"""
