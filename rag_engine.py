import json
import logging
from typing import List, Dict, Any, Optional

from openai import AzureOpenAI

logger = logging.getLogger(__name__)


# ── Summary generation (indexing phase) ────────────────────────────

def generate_ppt_summary(
    client: AzureOpenAI, deployment: str, ppt_data: Dict[str, Any]
) -> str:
    """Ask GPT-4o to produce a concise TOC / summary for one PPT."""
    slides_text = ""
    for slide in ppt_data["slides"]:
        slides_text += f"\n--- Slide {slide['slide_number']} ---\n{slide['content']}\n"

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document analyst. Create a concise summary and "
                    "table of contents for this PowerPoint presentation. Include "
                    "the main topics, key points, and what each section covers. "
                    "This summary will be used to determine if this presentation "
                    "is relevant to a user's question."
                ),
            },
            {
                "role": "user",
                "content": f"Presentation: {ppt_data['filename']}\n\n{slides_text}",
            },
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    return response.choices[0].message.content


# ── Stage 1 – find relevant PPTs ──────────────────────────────────

def find_relevant_ppts(
    client: AzureOpenAI,
    deployment: str,
    query: str,
    summaries: List[Dict[str, str]],
) -> List[str]:
    """Return filenames of PPTs relevant to the user's query."""
    summaries_block = ""
    for s in summaries:
        summaries_block += (
            f"\n\nFilename: {s['filename']}\nSummary: {s['summary']}\n"
        )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document routing assistant. Given a user's question "
                    "and summaries of available PowerPoint presentations, determine "
                    "which presentation(s) are relevant to answer the question.\n\n"
                    "Return ONLY a JSON array of relevant filenames.\n"
                    "If no presentation is relevant, return an empty array [].\n"
                    "Do not include any other text."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nAvailable presentations:{summaries_block}",
            },
        ],
        temperature=0.1,
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        logger.warning(f"Could not parse routing response as JSON: {raw}")

    # Fallback: check which filenames appear in the text
    return [s["filename"] for s in summaries if s["filename"] in raw]


# ── Stage 2 – generate grounded answer ─────────────────────────────

GROUNDING_SYSTEM_PROMPT = """You are a knowledge assistant. Answer the user's question \
using ONLY the provided PowerPoint slide content.

Rules:
1. Answer ONLY from the provided PPT content. Never use outside knowledge.
2. For every claim, cite the source as [Filename, Slide N].
3. If the answer cannot be found in the slides, respond with:
   "I couldn't find relevant information in the available presentations."
4. Be concise and accurate.
5. If information spans multiple presentations, combine the relevant parts and \
cite each source."""


def _build_context(ppt_contents: List[Dict[str, Any]]) -> str:
    """Concatenate slide content from one or more PPTs into a single string."""
    parts = []
    for ppt in ppt_contents:
        header = f"\n{'=' * 60}\nPresentation: {ppt['filename']}\n{'=' * 60}"
        slides = ""
        for slide in ppt["slides"]:
            slides += f"\n--- Slide {slide['slide_number']} ---\n{slide['content']}\n"
        parts.append(header + slides)
    return "".join(parts)


def generate_answer(
    client: AzureOpenAI,
    deployment: str,
    query: str,
    ppt_contents: List[Dict[str, Any]],
    *,
    stream: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1024,
):
    """Generate an answer grounded in the matched PPT content.

    Returns the raw OpenAI response (or streaming iterator when stream=True).
    """
    context = _build_context(ppt_contents)
    messages = [
        {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"PowerPoint Content:\n{context}\n\nQuestion: {query}",
        },
    ]

    return client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )


def generate_greeting_response(
    client: AzureOpenAI,
    deployment: str,
    query: str,
    topic_names: List[str],
    *,
    stream: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1024,
):
    """Respond to greetings with a friendly message listing available topics."""
    topic_list = "\n".join(f"  \u2022 {name}" for name in topic_names)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly knowledge assistant. The user has greeted you. "
                "Respond warmly and let them know what topics you can help with. "
                "Keep it concise and inviting."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User said: {query}\n\n"
                f"I have knowledge about the following presentations:\n{topic_list}\n\n"
                "Greet them back and list these topics so they know what to ask about."
            ),
        },
    ]

    return client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )


def generate_no_context_answer(
    client: AzureOpenAI,
    deployment: str,
    query: str,
    topic_names: Optional[List[str]] = None,
    *,
    stream: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1024,
):
    """Return a polite refusal when no PPT matches, listing available topics."""
    topic_info = ""
    if topic_names:
        topic_list = "\n".join(f"  \u2022 {name}" for name in topic_names)
        topic_info = (
            f"\n\nHowever, I have knowledge about the following topics:\n{topic_list}\n"
            "Let the user know what topics are available and invite them to ask "
            "questions related to these topics instead."
        )

    messages = [
        {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"No relevant presentations were found for this question.{topic_info}\n\n"
                f"Question: {query}"
            ),
        },
    ]

    return client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )
