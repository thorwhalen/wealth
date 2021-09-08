"""Utils"""

from py2store import LocalJsonStore

try:
    from importlib.resources import files  # ... and any other things you want to get
except ImportError:
    try:
        from importlib_resources import files  # pip install importlib_resources
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "No module named 'importlib_resources'. "
            'pip install importlib_resources or conda install importlib_resources'
        )

root_path = files('wealth')
data_dir = root_path / 'data'

json_store = LocalJsonStore(str(data_dir))

from datetime import datetime


def hms_message(msg=''):
    t = datetime.now()
    return '({:02.0f}){:02.0f}:{:02.0f}:{:02.0f} - {}'.format(
        t.day, t.hour, t.minute, t.second, msg
    )


def print_progress(msg, refresh=None, display_time=True):
    """
    input: message, and possibly args (to be placed in the message string, sprintf-style
    output: Displays the time (HH:MM:SS), and the message
    use: To be able to track processes (and the time they take)
    """
    if display_time:
        msg = hms_message(msg)
    if refresh:
        print(msg, end='\r')
        # stdout.write('\r' + msg)
        # stdout.write(refresh)
        # stdout.flush()
    else:
        print(msg)
