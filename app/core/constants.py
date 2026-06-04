"""Prompt templates and system constants."""


# --- System Prompts ---

GROUNDED_ANSWER_PROMPT = (
    "You are a strict grounded assistant for GlobalFreight Logistics.\n"
    "RULES:\n"
    "1. Answer ONLY from the provided context.\n"
    "2. If the context does not contain enough information, say exactly: "
    '"I don\'t know based on the provided context."\n'
    "3. Do NOT use outside knowledge.\n"
    "4. Do NOT invent or guess missing details.\n"
    "5. Do NOT answer if the context is irrelevant to the question.\n"
    "6. Cite the source document when possible.\n"
    "7. Keep answers concise and factual."
)

DIRECT_CHAT_PROMPT = (
    "You are a friendly, concise assistant.\n"
    "Use this path only for greetings, introductions, and casual small talk.\n"
    "Do not mention documents, retrieval, embeddings, or internal orchestration.\n"
    "Keep the answer short and natural."
)

VERIFICATION_PROMPT = (
    "You are a hallucination checker. Your job is to verify whether an answer "
    "is fully supported by the given context.\n"
    "Return ONLY valid JSON with these keys:\n"
    '  - "supported": boolean (true if all claims in the answer appear in context)\n'
    '  - "confidence": number between 0.0 and 1.0\n'
    '  - "reason": short string explaining your judgment\n'
    "If any important claim in the answer is NOT present in the context, "
    'set "supported" to false.\n'
    "Do not add any text outside the JSON object."
)

QUERY_REWRITE_PROMPT = (
    "Rewrite the user's question into a compact retrieval query.\n"
    "Return ONLY valid JSON with a single key: \"query\".\n"
    "Keep it short, specific, and keyword-rich.\n"
    "Do not answer the question."
)

SUMMARY_PROMPT = (
    "Summarize the following conversation into a brief, factual summary.\n"
    "Focus on: key topics discussed, important facts mentioned, "
    "and any user preferences noted.\n"
    "Keep it under 150 words. Do not add information not present in the conversation."
)


# --- Greeting Patterns ---

GREETING_PATTERNS = [
    r"^(hi|hello|hey|hii|helo|yo|sup)\b",
    r"^(good morning|good afternoon|good evening)\b",
    r"^(how are you|how r you)\b",
    r"^(who are you|what are you|what do you do|what can you do)\b",
    r"^(bye|goodbye|see you|take care)\b",
    r"^(thanks|thank you|thx)\b",
]


# --- Adversarial Patterns ---

ADVERSARIAL_PATTERNS = [
    r"ignore\s+(all\s+)?(previous\s+)?instructions",
    r"forget\s+(all\s+)?(previous\s+)?instructions",
    r"pretend\s+(the\s+)?documents?\s+say",
    r"hallucinate",
    r"make\s+up",
    r"generate\s+fake",
    r"act\s+as\s+if",
    r"override\s+(your\s+)?rules",
    r"disregard\s+(all\s+)?(safety|rules|guidelines)",
    r"jailbreak",
    r"DAN\s+mode",
]


# --- Response Constants ---

REFUSAL_RESPONSE = "I don't know based on the provided context."

IDK_VARIANTS = [
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i cannot answer",
    "insufficient information",
    "not enough context",
    "no relevant information",
]

# --- Embedding Task Prefix ---
EMBED_QUERY_PREFIX = "task: question answering | query: "
EMBED_DOC_PREFIX = "task: question answering | query: "
