"""
Turning uploaded files into model input.

Each modality gets its own loader, and both share one rule: never guess.
A missing pixel spacing is reported as missing rather than assumed to be
1 mm; an ECG with the wrong number of leads is rejected rather than padded.
"""
