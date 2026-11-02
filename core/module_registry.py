import importlib
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path

from core.base_module import BaseModule

log = logging.getLogger(__name__)

MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"


@dataclass
class ModuleEntry:
    package: str
    folder: Path
    cls: type[BaseModule] | None = None
    error: str | None = None  # set when the module failed to load


def discover_modules() -> list[ModuleEntry]:
    """Import every modules/<package>/module.py and collect its BaseModule subclasses.

    A module that fails to import (or defines no module class) is returned as an
    entry with `error` set, so the hub can show a disabled card instead of crashing.
    """
    entries = []
    for folder in sorted(MODULES_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith(("_", ".")):
            continue
        if not (folder / "module.py").is_file():
            continue

        package = folder.name
        try:
            mod = importlib.import_module(f"modules.{package}.module")
        except Exception as exc:
            log.exception("Failed to import module %r", package)
            entries.append(ModuleEntry(package, folder, error=f"{type(exc).__name__}: {exc}"))
            continue

        classes = [
            obj
            for _, obj in inspect.getmembers(mod, inspect.isclass)
            if issubclass(obj, BaseModule)
            and obj.__module__ == mod.__name__
            and not inspect.isabstract(obj)
        ]
        if not classes:
            log.error("Module %r defines no BaseModule subclass", package)
            entries.append(ModuleEntry(package, folder, error="No BaseModule subclass found"))
            continue

        for cls in classes:
            entries.append(ModuleEntry(package, folder, cls=cls))

    return entries
