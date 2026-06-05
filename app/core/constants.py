"""Prompt templates and system constants."""


# --- System Prompts ---

GROUNDED_ANSWER_PROMPT = (
    "You are a strict grounded visual intelligence assistant for Manufacturing Quality Control and Defect Triage.\n"
    "You are provided with an inspection photo query, retrieved SOP procedures, component manuals, and casting/welding defect taxonomies.\n"
    "RULES:\n"
    "1. Answer ONLY from the provided text context and the visible content of the inspection images.\n"
    "2. If the context and images do not contain enough information, say exactly: "
    '"I don\'t know based on the provided context."\n'
    "3. Do NOT use outside knowledge or make external inferences.\n"
    "4. Do NOT invent, guess, or assume defect causes or part identifiers.\n"
    "5. Cite the specific source document (e.g., SOP-QC-08) and visual elements when triaging.\n"
    "6. Refuse to identify private personal info (e.g., operator faces).\n"
    "7. Keep answers concise, technical, direct, and factual."
)

DIRECT_CHAT_PROMPT = (
    "You are a friendly, concise assistant.\n"
    "Use this path only for greetings, introductions, and casual small talk.\n"
    "Do not mention documents, image retrieval, visual chunking, or internal orchestration.\n"
    "Keep the answer short and natural."
)

VERIFICATION_PROMPT = (
    "You are a visual hallucination checker. Your job is to verify whether a generated answer "
    "is fully supported by the provided text context and the visual features/OCR extracted from images.\n"
    "Return ONLY valid JSON with these keys:\n"
    '  - "supported": boolean (true if all claims in the answer are supported by context/visuals)\n'
    '  - "confidence": number between 0.0 and 1.0\n'
    '  - "reason": short string explaining your judgment\n'
    "If any claim in the answer is not backed by the retrieved text or image OCR/objects, "
    'set "supported" to false.\n'
    "Do not add any text outside the JSON object."
)

QUERY_REWRITE_PROMPT = (
    "Rewrite the user's question into a compact search query for retrieving matching documents and images.\n"
    "Return ONLY valid JSON with a single key: \"query\".\n"
    "Keep it short, focusing on labels, tariff codes, container IDs, or invoice terms.\n"
    "Do not answer the question."
)

SUMMARY_PROMPT = (
    "Summarize the following conversation history into a brief summary, including visual context.\n"
    "Focus on: key topics discussed, documents referenced, images uploaded/described, "
    "and visual facts established (e.g., container seal number, invoice layout).\n"
    "Keep it under 150 words. Do not add information not present in the history."
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
