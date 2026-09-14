"""Entry point: ``python -m news_qa``."""

import sys


def _use_utf8_console():
    """Keep Persian output readable when stdout is not already UTF-8.

    On Windows a redirected stream defaults to the legacy code page, which
    raises UnicodeEncodeError on the first Persian character. Reconfiguring
    is a no-op everywhere else.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass


def main():
    _use_utf8_console()

    from .cli import main as run
    return run()


if __name__ == '__main__':
    sys.exit(main())
