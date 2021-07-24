"""Utils"""


try:
    from importlib.resources import files  # ... and any other things you want to get
except ImportError:
    try:
        from importlib_resources import files  # pip install importlib_resources
    except ModuleNotFoundError:
        raise ModuleNotFoundError("No module named 'importlib_resources'. "
                                  "pip install importlib_resources or conda install importlib_resources")

root_path = files('wealth')
data_dir = root_path / 'data'
