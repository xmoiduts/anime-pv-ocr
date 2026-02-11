"""
Shared state selection logic for comparison video panels.
"""

import bisect
from typing import List, Optional


def get_active_index_at_or_before(
    sorted_frame_ids: List[int],
    current_frame_id: int,
    fallback_to_first: bool = True,
) -> Optional[int]:
    """
    Pick index of nearest frame id <= current_frame_id.

    If there is no frame id <= current_frame_id, returns 0 when
    fallback_to_first is True, otherwise returns None.
    """
    if not sorted_frame_ids:
        return None

    pos = bisect.bisect_right(sorted_frame_ids, current_frame_id) - 1
    if pos >= 0:
        return pos
    return 0 if fallback_to_first else None

