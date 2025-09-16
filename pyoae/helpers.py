"""Module with general helper functions for PyOAE."""

import re


def sanitize_filename_part(part: str) -> str:
    """Remove characters invalid in filenames and strip whitespace."""
    # Invalid on Windows: \ / : * ? " < > |
    return re.sub(r'[\\/:*?"<>|]', "", part).strip()
