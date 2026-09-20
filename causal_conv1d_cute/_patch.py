"""Replace a module-level function everywhere it has been imported by name."""

import functools
import importlib
import sys

_MARK = "_causal_conv1d_cute"


def wrap(target, around):
    """Wrap the function at a dotted path and rebind every loaded module that imported it.

    A module that ran ``from package import function`` holds its own reference to the function, so
    replacing the attribute of the defining module alone would leave that reference in place. Every
    loaded module is therefore scanned for attributes that are the original function object.

    Args:
        target: Dotted path of a module-level function, such as "package.module.function".
        around: Callable invoked as around(original, *args, **kwargs) in place of the original.

    Returns:
        The number of module attributes that were rebound; zero if the function was already
        wrapped.
    """
    module_name, name = target.rsplit(".", 1)
    original = getattr(importlib.import_module(module_name), name)
    if getattr(original, _MARK, False):
        return 0
    wrapped = functools.wraps(original)(functools.partial(around, original))
    setattr(wrapped, _MARK, True)
    count = 0
    for module in list(sys.modules.values()):
        try:
            items = list(vars(module).items())
        except TypeError:
            continue
        for attr, value in items:
            if value is original:
                setattr(module, attr, wrapped)
                count += 1
    return count
