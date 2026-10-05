"""The Zhouyi APIs these probes use, from whichever tinygrad tree `TG` points at.

The current backend (the `tinygrad/` submodule) keeps only the device under `tinygrad/` and has the custom ops, hand-written
kernels and the AIFF engine under `extra/zhouyi/`; older trees have everything under `tinygrad/runtime/`. Import after the
script has put its tinygrad tree on `sys.path`:

    from zy import OA                       # the custom ops (+ the device module's names): OA.gemm_gs, OA.csrc_call, OA.Device ...
    from zy import vec_f16 as V, gemm_fp16  # any Zhouyi module, wherever it lives
"""
import importlib


def _module(name: str):
  for base in ("extra.zhouyi", "tinygrad.runtime.support.zhouyi"):
    try: return importlib.import_module(f"{base}.{name}")
    except ModuleNotFoundError as e:
      if e.name not in (f"{base}.{name}", base, base.split(".")[0]): raise
  raise ModuleNotFoundError(f"no Zhouyi module {name!r} in extra.zhouyi or tinygrad.runtime.support.zhouyi")


class _Ops:
  """`extra.zhouyi.ops` (importing it registers the custom ops) first, then `tinygrad.runtime.ops_zhouyi`."""
  def __init__(self): self._mods = None
  def _load(self):
    if self._mods is None:
      mods = []
      try: mods.append(importlib.import_module("extra.zhouyi.ops"))
      except ModuleNotFoundError as e:
        if e.name not in ("extra.zhouyi.ops", "extra.zhouyi", "extra"): raise
      mods.append(importlib.import_module("tinygrad.runtime.ops_zhouyi"))
      self._mods = mods
    return self._mods
  def __getattr__(self, name):
    if name.startswith("__"): raise AttributeError(name)
    for m in self._load():
      if hasattr(m, name): return getattr(m, name)
    raise AttributeError(f"no Zhouyi op {name!r}")


OA = _Ops()


def __getattr__(name):
  if name.startswith("__"): raise AttributeError(name)
  return _module(name)
