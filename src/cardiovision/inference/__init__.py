"""
The trained models.

Named ``inference`` rather than ``models`` on purpose: ``models/`` at the repo
root holds the weights, and giving a package the same name makes every
traceback ambiguous about which one you are looking at.

Each module owns one model, exposes a single module-level instance, and loads
lazily so that one broken checkpoint cannot take the service down with it.
"""
