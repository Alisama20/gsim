"""Debug-only pytest plugin: trace the active PDK and the meep layer resolution.

Loaded with `-p flaky_probe` from .github/debug. Writes flaky_probe.log.
"""

from __future__ import annotations

import json
import os
import traceback

LOG = os.environ.get("FLAKY_PROBE_LOG", "flaky_probe.log")
VICTIMS = (
    "test_real_shallow_etch_stays_separate_from_direct_slab",
    "test_direct_slab_does_not_create_shallow_etch",
    "test_build_config_and_xz_geometry_omit_false_shallow_etch",
)
_last = {"pdk": None}


def _write(line: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _pdk_name() -> str:
    try:
        import gdsfactory as gf

        p = gf.get_active_pdk()
        return f"{p.name}@{id(p):x}"
    except Exception as e:  # noqa: BLE001
        return f"<error {e!r}>"


def _dump(tag: str, nodeid: str) -> None:
    info: dict = {"tag": tag, "test": nodeid, "pdk": _pdk_name()}
    try:
        import gdsfactory as gf

        from gsim.common.stack import get_stack
        from gsim.meep import physical_layers as pl

        stack = get_stack()
        info["stack_pdk"] = getattr(stack, "pdk_name", None)
        src = pl._active_gdsfactory_stack(stack)
        info["source_stack"] = src is not None
        for name in ("core", "slab150", "shallow_etch"):
            layer = stack.layers.get(name)
            if layer is None:
                info[name] = None
                continue
            level = pl._matching_layer_level(name, layer, src) if src is not None else None
            info[name] = {"gds_layer": str(layer.gds_layer), "level": level is not None}
        try:
            layers = gf.get_active_pdk().layers
            info["pdk_SLAB150"] = str(getattr(layers, "SLAB150", None))
            info["pdk_WG"] = str(getattr(layers, "WG", None))
        except Exception as e:  # noqa: BLE001
            info["pdk_layers_error"] = repr(e)
    except Exception:  # noqa: BLE001
        info["error"] = traceback.format_exc(limit=3)
    _write(json.dumps(info, default=str))


def pytest_sessionstart(session):  # noqa: ARG001
    _write(json.dumps({"tag": "start", "pdk": _pdk_name()}))


def pytest_runtest_setup(item):
    if item.name.split("[")[0] in VICTIMS:
        _dump("victim_setup", item.nodeid)


def pytest_runtest_teardown(item, nextitem):  # noqa: ARG001
    with open("test_order.txt", "a", encoding="utf-8") as f:
        f.write(item.nodeid + chr(10))
    name = _pdk_name()
    if name != _last["pdk"]:
        _write(json.dumps({"tag": "pdk_change_after", "test": item.nodeid, "pdk": name}))
        _last["pdk"] = name


def pytest_runtest_logreport(report):
    if report.when == "call" and report.failed and report.nodeid.split("::")[-1].split("[")[0] in VICTIMS:
        _write(json.dumps({"tag": "victim_failed", "test": report.nodeid, "longrepr": str(report.longrepr)[-2000:]}))
