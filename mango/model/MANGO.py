"""Compatibility stub for the retired package-level model.

The maintained implementation is ``workflow/scripts/model_common.py``. It
constructs an antigen-only prefix and validates checkpoint architecture before
training or inference.
"""


class MANGO:
    """Compatibility name that prevents accidental use of the retired model."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "mango.model.MANGO is retired. Use workflow/scripts/model_common.py "
            "through the supported Snakemake workflow."
        )
