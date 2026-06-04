import os
import sys
import asyncio
import gradio as gr
from pathlib import Path

# Ensure project root is in path
sys.path.append(str(Path(__file__).parent.absolute()))

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("gradio_app")

# Global RAG orchestrator
orchestrator = None

async def init_rag():
    global orchestrator
    settings = get_settings()
    logger.info("Initializing RAG pipeline for Hugging Face Space...")

    # Set up cache backend (fallback to InMemory)
    from app.cache.redis_client import get_cache_backend
    cache_backend = await get_cache_backend()

    from app.memory.conversation import ConversationManager
    from app.memory.summary import SummaryManager
    conversation_manager = ConversationManager(cache_backend)
    summary_manager = SummaryManager(cache_backend)

    # Ingest documents on startup (so the Space has data immediately)
    from app.main import run_ingestion
    try:
        logger.info("Ingesting source documents on startup...")
        result = await run_ingestion(
            source_dir="rag_docs/rag_docs",
            force_reindex=True
        )
        logger.info("Ingestion complete!", documents=result["documents_loaded"], chunks=result["chunks_indexed"])
    except Exception as e:
        logger.exception("Document ingestion failed during startup", error=str(e))

    # Retrieve components from app.main state
    import app.main
    orchestrator = app.main.get_orchestrator()

# Run initialization
try:
    # Ensure event loop runs cleanly
    asyncio.run(init_rag())
except Exception as e:
    logger.exception("Error running RAG initialization on startup", error=str(e))


async def predict(message, history, username):
    global orchestrator
    if not orchestrator:
        return "System is not initialized. Please verify your environment and API keys.", {}

    if not username:
        username = "hf_user"

    try:
        # Process query through RAG pipeline
        response = await orchestrator.process_query(
            question=message,
            username=username,
            session_id=f"session_{username}"
        )

        # Build detailed trace metadata
        metadata = {
            "Trace ID": response.get("trace_id", ""),
            "Routing Mode": response.get("mode", ""),
            "Grounded Confidence": response.get("confidence", 0.0),
            "Cached": response.get("cached", False),
            "Timestamp": response.get("timestamp", ""),
            "Verifier Verdict": {
                "Supported": response.get("verdict", {}).get("supported", False) if response.get("verdict") is not None else False,
                "Confidence": response.get("verdict", {}).get("confidence", 0.0) if response.get("verdict") is not None else 0.0,
                "Reasoning": response.get("verdict", {}).get("reason", "") if response.get("verdict") is not None else ""
            },
            "Token Usage": {
                "Prompt Tokens": response.get("usage", {}).get("prompt_tokens", 0) if response.get("usage") is not None else 0,
                "Completion Tokens": response.get("usage", {}).get("completion_tokens", 0) if response.get("usage") is not None else 0,
                "Total Tokens": response.get("usage", {}).get("total_tokens", 0) if response.get("usage") is not None else 0
            },
            "Retrieved Sources": [
                {
                    "Source": s.get("source", "unknown") if s else "unknown",
                    "Chunk ID": s.get("chunk_id", 0) if s else 0,
                    "Score": s.get("score", 0.0) if s else 0.0,
                    "Preview": s.get("text_preview", "") if s else ""
                } for s in response.get("sources", []) if isinstance(s, dict)
            ] if response.get("sources") is not None else []
        }

        return response["answer"], metadata
    except Exception as e:
        logger.exception("Error processing chat message", error=str(e))
        return f"An error occurred: {str(e)}", {"error": str(e)}


def make_ui():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # 🚢 PSI RAG: Production Guardrailed Self-RAG
            ### GlobalFreight Logistics Document QA Assistant
            
            This assistant is grounded in carrier SLA agreements, customs tariffs, and shipment delay policies.
            It uses a 10-layer guardrail stack to prevent jailbreaks, small-talk routing, hallucination verification, and multi-level caching.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat History")
                msg = gr.Textbox(
                    label="Ask a question about SLA, delay policies, or customs tariffs...",
                    placeholder="What is the transit time for Delhi to New York under Gold standard?"
                )
                username = gr.Textbox(label="Username (for session isolation & memory cache)", value="guest_user")
                clear = gr.ClearButton([msg, chatbot])
            
            with gr.Column(scale=2):
                gr.Markdown("### 🔍 Live Request Trace")
                trace_json = gr.JSON(label="Metadata & Token Usage")
                sources_md = gr.Markdown(label="Retrieved Source Chunks")
        
        # When user submits message
        async def user_respond(message, chat_history, user):
            if chat_history is None:
                chat_history = []
            bot_msg, trace = await predict(message, chat_history, user)
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_msg})
            
            # Format sources markdown
            sources = trace.get("Retrieved Sources", [])
            sources_text = "#### Retrieved Context:\n"
            if not sources:
                sources_text += "*No document chunks retrieved for this mode (direct smalltalk/refusal)*"
            else:
                for idx, src in enumerate(sources):
                    sources_text += f"**Chunk {idx+1} ({src['Source']})** - Similarity: `{src['Score']:.3f}`\n"
                    sources_text += f"> *{src['Preview']}...*\n\n"
            
            return "", chat_history, trace, sources_text
            
        msg.submit(user_respond, [msg, chatbot, username], [msg, chatbot, trace_json, sources_md])
        
    return demo


demo = make_ui()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"))
