from __future__ import annotations

import re


SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b", re.IGNORECASE)


def canonicalize_player_name(name: str) -> str:
    text = re.sub(r"\s+", " ", str(name or "").strip())
    if not text:
        return ""
    if "," in text:
        last, first = [part.strip() for part in text.split(",", 1)]
        if first and last:
            text = f"{first} {last}"
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    text = SUFFIX_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def dfs_name_key(name: str) -> str:
    canonical = canonicalize_player_name(name)
    if not canonical:
        return ""
    parts = canonical.split()
    collapsed: list[str] = []
    i = 0
    while i < len(parts):
        if len(parts[i]) == 1:
            initials = [parts[i]]
            j = i + 1
            while j < len(parts) and len(parts[j]) == 1:
                initials.append(parts[j])
                j += 1
            collapsed.append("".join(initials))
            i = j
            continue
        collapsed.append(parts[i])
        i += 1
    return " ".join(collapsed)
