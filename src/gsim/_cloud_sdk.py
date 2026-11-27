"""Load the cloud client (``gdsfactoryplus.sim``) when it is first needed.

``import gsim.palace`` (and every other solver module) must work on a machine
that has no cloud client: meshing and local runs do not use one. The client is
therefore not imported at module import time. ``sim`` below stands in for the
module ``gdsfactoryplus.sim``: the first attribute access imports it, and a
missing client then raises :class:`CloudClientNotInstalledError`, which says
what to install.

Reads, writes and deletes of attributes go to the real module, so patching
``gsim.gcloud.sim.<name>`` acts on ``gdsfactoryplus.sim`` as it does when the
module is imported directly.
"""

from __future__ import annotations

from typing import Any


class CloudClientNotInstalledError(ImportError):
    """The cloud client ``gdsfactoryplus.sim`` cannot be imported."""


def load_sim() -> Any:
    """Import and return the module ``gdsfactoryplus.sim``.

    Raises:
        CloudClientNotInstalledError: If the module is not importable. The
            message names the package and says that only the cloud steps need it.
    """
    try:
        from gdsfactoryplus import sim
    except ImportError as error:
        raise CloudClientNotInstalledError(
            "Cloud simulation needs the GDSFactory+ client, the module "
            "'gdsfactoryplus.sim', and it cannot be imported "
            f"({error}). Install or update the 'gdsfactoryplus' package and "
            "sign in. Meshing, config generation and local runs do not need it."
        ) from error
    return sim


class _LazySim:
    """Stands in for the module ``gdsfactoryplus.sim``; see the module docstring."""

    @property
    def __dict__(self) -> dict[str, Any]:  # type: ignore[override]
        """The namespace of ``gdsfactoryplus.sim``.

        ``unittest.mock.patch`` reads ``target.__dict__`` to tell an attribute
        the target owns from an inherited one, and restores or deletes it on
        that basis. Answering with the module's namespace makes a patch on this
        object undo exactly as a patch on the module itself does.
        """
        return vars(load_sim())

    def __dir__(self) -> list[str]:
        """List the names of ``gdsfactoryplus.sim``."""
        return dir(load_sim())

    def __getattr__(self, name: str) -> Any:
        """Read *name* from ``gdsfactoryplus.sim``, importing it on first use."""
        # Only called for names this object lacks. Dunder lookups (copy,
        # pickle, inspect) must not trigger an import.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(load_sim(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set *name* on ``gdsfactoryplus.sim``."""
        setattr(load_sim(), name, value)

    def __delattr__(self, name: str) -> None:
        """Delete *name* from ``gdsfactoryplus.sim``."""
        delattr(load_sim(), name)

    def __repr__(self) -> str:
        """Name the module this object stands for."""
        return "<lazy gdsfactoryplus.sim>"


sim: Any = _LazySim()

__all__ = ["CloudClientNotInstalledError", "load_sim", "sim"]
