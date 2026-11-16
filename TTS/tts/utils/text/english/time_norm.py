import re

from TTS.tts.utils.text.english.number_norm import _expand_num

_time_re = re.compile(
    r"""\b
                          ((0?[0-9])|(1[0-1])|(1[2-9])|(2[0-3]))  # hours
                          :
                          ([0-5][0-9])(?!\d)                      # minutes
                          (?:\s*(?P<am_pm>[ap])(?:m\b|\.m\b(?P<dot>\.)?))?  # am, a.m, a.m., pm, ...
                          (?!\w)""",
    re.IGNORECASE | re.VERBOSE,
)


def _expand_time_english(match: re.Match) -> str:
    hour = int(match.group(1))
    past_noon = hour >= 12
    time = []
    if hour > 12:
        hour -= 12
    elif hour == 0:
        # 00:xx is midnight, so it stays a.m. 12:xx is already past noon.
        hour = 12
    time.append(_expand_num(hour))

    minute = int(match.group(6))
    if minute > 0:
        if minute < 10:
            time.append("oh")
        time.append(_expand_num(minute))
    am_pm = match.group("am_pm")
    if am_pm is None:
        time.append("p m" if past_noon else "a m")
    else:
        time.extend([am_pm.lower(), "m"])
    text = " ".join(time)
    # Drop the final dot of "a.m." mid-sentence, but keep it when it ends the text,
    # where it doubles as the sentence-final period.
    if match.group("dot") and not match.string[match.end() :].strip():
        text += "."
    return text


def expand_time_english(text: str) -> str:
    return re.sub(_time_re, _expand_time_english, text)
