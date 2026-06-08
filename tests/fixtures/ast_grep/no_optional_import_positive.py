try:
    import psutil
except ImportError:
    psutil = None

try:
    from optional_package import feature
except ModuleNotFoundError as exc:
    feature = None
