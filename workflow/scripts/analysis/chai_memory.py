"""Delay Chai's later-stage weights until their first GPU forward pass."""

from contextlib import contextmanager


class DeferredModule:
    def __init__(self, loader, key, device):
        self.loader = loader
        self.key = key
        self.device = device
        self.module = None

    def forward(self, *args, **kwargs):
        if self.module is None:
            self.module = self.loader(self.key, self.device)
        return self.module.forward(*args, **kwargs)


@contextmanager
def defer_later_stage_weights(chai_module):
    """Preserve the upstream loader and computation, reducing trunk residency."""
    original = chai_module.load_exported

    def load(key, device):
        if key in {"diffusion_module.pt", "confidence_head.pt"}:
            return DeferredModule(original, key, device)
        return original(key, device)

    chai_module.load_exported = load
    try:
        yield
    finally:
        chai_module.load_exported = original
