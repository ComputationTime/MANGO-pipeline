"""Retired pre-workflow runner.

The supported model requires a trained workflow checkpoint and a precomputed
antigen embedding. Use ``workflow/scripts/generate_designs.py`` or the Snakemake
``generate`` target so the antigen-only contract is enforced.
"""


class MANGORunner:
    """Compatibility name that fails instead of invoking an obsolete model."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "MANGORunner is retired. Use the Snakemake workflow or "
            "workflow/scripts/generate_designs.py with an antigen embedding."
        )
