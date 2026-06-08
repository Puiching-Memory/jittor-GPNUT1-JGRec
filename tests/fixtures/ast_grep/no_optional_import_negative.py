import psutil


try:
    value = compute_value()
except ImportError:
    value = None

try:
    import required_package
except RuntimeError:
    recover()
