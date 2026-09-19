"""Draft a reply to a ticket, using only what a similar past ticket tells us.

A small model asked to write a reply from scratch tends to ignore the facts, refuse, or invent
things. So the model gets a narrow job and three safeguards surround it:

1. Similarity threshold (Module 04): if no past ticket is close enough, escalate to a human.
2. Narrow task (Module 02): the model only restates ONE known resolution politely.
3. Grounding checks: the draft must reuse the resolution's key words AND add almost nothing of its
   own. If either check fails, use a plain template instead.

These are simple word-overlap heuristics, not proof of correctness. They catch obvious drift; they
cannot tell whether the resolution actually answers the customer's question (that is what the
similarity score and a human are for). Module 08 covers stronger guardrails.
"""

import re

MIN_SIMILARITY = 0.35  # chosen by hand: off-topic questions scored below 0.1 in Module 04, real ones above 0.3. Not tuned.
MIN_GROUNDING = 0.4  # share of the resolution's key words the draft must reuse
MAX_NOVELTY = 0.35  # share of the draft's words allowed to come from nowhere (ticket, resolution, politeness)

ESCALATE_REPLY = "Thanks for contacting us. This one needs a specialist, so we have passed it on and will follow up soon."

REPLY_SYSTEM_PROMPT = (
    "You write short, polite customer support replies of at most 2 sentences.\n"
    "Restate the resolution you are given so the customer understands it. Add no other facts."
)

STOPWORDS = set("a an and are as at be by for from in is it of on or that the to was we will with you your".split())
POLITE_WORDS = set("sorry hear thanks thank reaching please let know help assist assistance further feel free reach again apologize".split())


def content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def grounding_score(draft, resolution):
    """Fraction of the resolution's key words that the draft reuses (0 to 1)."""
    needed = content_words(resolution)
    return len(needed & content_words(draft)) / len(needed) if needed else 0.0


def novelty_score(draft, resolution, ticket_text):
    """Fraction of the draft's key words that appear in neither the ticket, the resolution, nor polite filler."""
    allowed = content_words(resolution) | content_words(ticket_text) | POLITE_WORDS
    words = content_words(draft)
    return len(words - allowed) / len(words) if words else 1.0


def build_reply_messages(text, resolution):
    user = f"Customer wrote: {text}\nResolution: {resolution}\n\nWrite the reply."
    return [{"role": "system", "content": REPLY_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def draft_reply(text, neighbors, chat_fn):
    """Return (reply, source) where source is 'model', 'template', or 'escalate'."""
    best_ticket, similarity = neighbors[0]
    if similarity < MIN_SIMILARITY:
        return ESCALATE_REPLY, "escalate"

    resolution = best_ticket["resolution"]
    draft = chat_fn(build_reply_messages(text, resolution), max_new_tokens=80)
    grounded = grounding_score(draft, resolution) >= MIN_GROUNDING
    stays_on_topic = novelty_score(draft, resolution, text) <= MAX_NOVELTY
    if grounded and stays_on_topic:
        return draft, "model"
    return f"Thanks for reaching out. {resolution}", "template"
