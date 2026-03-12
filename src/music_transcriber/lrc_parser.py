import re


LRC_FENCE_RE = re.compile(r"```(?:lrc|LRC)\s*\n(.*?)\n```", re.DOTALL)
GENERIC_FENCE_RE = re.compile(r"```\s*\n(.*?)\n```", re.DOTALL)
LRC_LINE_RE = re.compile(r"^\[(?:\d{2}:\d{2}(?:\.\d{1,3})?|ar:[^\]]+|ti:[^\]]+|al:[^\]]+|by:[^\]]+|offset:-?\d+)\].*$")


def _normalize_lrc_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def _extract_candidate_lines(text: str) -> str:
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if line and LRC_LINE_RE.match(line):
            lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def extract_lrc_text(response_text: str) -> str:
    if not response_text or not response_text.strip():
        raise ValueError("Gemini response is empty.")

    match = LRC_FENCE_RE.search(response_text)
    if match:
        normalized = _normalize_lrc_text(match.group(1))
        if normalized.strip():
            return normalized

    match = GENERIC_FENCE_RE.search(response_text)
    if match:
        normalized = _normalize_lrc_text(match.group(1))
        candidates = _extract_candidate_lines(normalized)
        if candidates.strip():
            return candidates

    fallback = _extract_candidate_lines(response_text)
    if fallback.strip():
        return fallback

    raise ValueError("Could not extract an LRC block from the Gemini response.")
