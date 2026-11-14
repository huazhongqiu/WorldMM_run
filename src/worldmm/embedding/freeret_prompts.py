"""Pure templates copied from VideoSpy videospy/model/prompts.py (2026-11-14).

Kept local so WorldMM does not acquire a VideoSpy runtime dependency.
"""

IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

QUERY_SEARCH_TEMPLATE = """Task: Represent a text query for retrieving matching short video clips.

Query: {query}

Based only on the query, output exactly one word following these rules:
- Capture the core semantics of the event, action, object, scene, state, or spoken topic to retrieve.
- Do not use function words, prepositions, symbols, or incomplete fragments."""

CLIP_SEARCH_TEMPLATE = """Task: Represent a short video clip for retrieval by text queries.

Video frames in chronological order:
{frames}

{subtitle_section}Based only on the supplied frames and subtitles, output exactly one word following these rules:
- Capture the core semantics of the most distinctive event, action, object, scene, state, or spoken topic in this clip.
- Use frame order to distinguish actions and state changes. Spoken text does not necessarily describe visible events.
- Do not explain or infer unseen events.
- Do not use function words, prepositions, symbols, or incomplete fragments."""


def query_prompt(query: str) -> str:
    return QUERY_SEARCH_TEMPLATE.format(query=query)


def _clip_evidence(timestamps, subtitles, start_time):
    if any(later < earlier for earlier, later in zip(timestamps, timestamps[1:])):
        raise ValueError("Frame timestamps must be chronological.")
    frames = "\n".join(
        f"[+{timestamp - start_time:.3f}s] {IMAGE_TOKEN}" for timestamp in timestamps
    )
    subtitle_section = ""
    if subtitles:
        lines = "\n".join(
            f"[+{max(0, item['start_time'] - start_time):.3f}s to "
            f"+{item['end_time'] - start_time:.3f}s] {item['subtitle']}"
            for item in subtitles
        )
        subtitle_section = f"Subtitles overlapping this clip:\n{lines}\n\n"
    return frames, subtitle_section


def clip_prompt(timestamps, subtitles, start_time) -> str:
    frames, subtitle_section = _clip_evidence(timestamps, subtitles, start_time)
    return CLIP_SEARCH_TEMPLATE.format(frames=frames, subtitle_section=subtitle_section)

TEXT_SEARCH_TEMPLATE = """Task: Represent a text passage for semantic retrieval.

Text: {text}

Output exactly one word that captures its central meaning. Do not explain."""


def text_prompt(text):
    return TEXT_SEARCH_TEMPLATE.format(text=text)
