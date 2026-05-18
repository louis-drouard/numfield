"""
Utility functions for field representation and formatting.

This module provides helper functions for converting Python slicing notation
into human-readable string representations, useful for debugging and logging.

Examples
--------
>>> from field.utils import slice_to_str
>>> slice_to_str(slice(1, 5))
'[1:5]'
>>> slice_to_str((slice(None), 0))
'[:, 0]'
"""

def slice_to_str(slc):
    """Convert slice tuple (ex: arr[1:,:]) to string '[:, :, 0]'."""
    def one_slice(s):
        if isinstance(s, slice):
            start = '' if s.start is None else s.start
            stop = '' if s.stop is None else s.stop
            step = '' if s.step is None else s.step
            if step == '':
                return f"{start}:{stop}"
            else:
                return f"{start}:{stop}:{step}"
        else:
            return str(s)

    # Si c'est un seul slice, pas un tuple
    if not isinstance(slc, tuple):
        slc = (slc,)
    
    return "[" + ", ".join(one_slice(s) for s in slc) + "]"