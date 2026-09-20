"""Reproducible ICLR revision experiments.

The modules in this package are orchestration and audit utilities. They do not
change the headline training/evaluation definitions; every expensive command
is emitted by the shell protocols with an explicit base model, seed, split
hash, and isolated output directory.
"""

PROTOCOL_VERSION = "iclr2027_revision_v1"

