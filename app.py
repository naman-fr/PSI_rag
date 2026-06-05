import os
import sys
import asyncio
import gradio as gr
from pathlib import Path
from PIL import Image

# Ensure project root is in path
sys.path.append(str(Path(__file__).parent.absolute()))

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("gradio_app")

# Global RAG orchestrator
orchestrator = None
init_error = None

async def init_rag():
    global orchestrator, init_error
    settings = get_settings()
    logger.info("Initializing Multimodal CV-RAG pipeline for Hugging Face Space...")

    # Set up cache backend (fallback to InMemory)
    from app.cache.redis_client import get_cache_backend
    cache_backend = await get_cache_backend()

    from app.memory.conversation import ConversationManager
    from app.memory.summary import SummaryManager
    conversation_manager = ConversationManager(cache_backend)
    summary_manager = SummaryManager(cache_backend)

    import app.main
    app.main._cache_backend = cache_backend
    app.main._conversation_manager = conversation_manager
    app.main._summary_manager = summary_manager

    # Ingest document text on startup
    from app.main import run_ingestion
    try:
        logger.info("Ingesting source logistics documents on startup...")
        result = await run_ingestion(
            source_dir="rag_docs/rag_docs",
            force_reindex=True
        )
        logger.info("Text documents loaded successfully!", documents=result["documents_loaded"], chunks=result["chunks_indexed"])
    except Exception as e:
        import traceback
        init_error = f"Ingestion Exception: {str(e)}\n{traceback.format_exc()}"
        logger.exception("Document ingestion failed during startup", error=str(e))
        return

    # Retrieve components from app.main state
    import app.main
    orchestrator = app.main.get_orchestrator()
    if not orchestrator:
        init_error = "Orchestrator retrieved from app.main is None"

# Run initialization
try:
    asyncio.run(init_rag())
except Exception as e:
    import traceback
    init_error = f"Startup Event Loop Exception: {str(e)}\n{traceback.format_exc()}"
    logger.exception("Error running RAG initialization on startup", error=str(e))


async def predict(message, image_path, history, username):
    global orchestrator, init_error
    if not orchestrator:
        err_msg = f"System is not initialized. Please verify your environment and API keys.\n\nInitialization Error:\n{init_error or 'Unknown Error'}"
        return err_msg, {}

    if not username:
        username = "hf_user"

    raw_image = None
    if image_path:
        try:
            raw_image = Image.open(image_path)
            logger.info("Opened user uploaded image for VQA", path=image_path, size=raw_image.size)
        except Exception as e:
            logger.error("Failed to open uploaded image", path=image_path, error=str(e))
            return f"Error opening uploaded image: {str(e)}", {}

    try:
        # Process query through multimodal RAG pipeline
        response = await orchestrator.process_query(
            question=message or "Describe this image and list its key fields.",
            username=username,
            session_id=f"session_{username}",
            raw_image=raw_image
        )

        # Retrieve cached features for trace info if available
        cached_ocr = ""
        cached_tags = []
        if raw_image and hasattr(orchestrator, "image_feature_cache"):
            # Image ID is derived or generated in process_query. We can check metadata in response
            # Or get it from the trace details.
            pass

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
                    "Chunk/Tile ID": s.get("chunk_id", 0) if s else 0,
                    "Score": s.get("score", 0.0) if s else 0.0,
                    "Preview": s.get("text_preview", "") if s else ""
                } for s in response.get("sources", []) if isinstance(s, dict)
            ] if response.get("sources") is not None else []
        }

        return response["answer"], metadata
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.exception("Error processing chat message", error=str(e))
        return f"An error occurred: {str(e)}\n\nTraceback:\n{tb}", {"error": str(e), "traceback": tb}


def make_ui():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # 🚢 GlobalFreight Logistics: Multimodal CV-RAG
            ### Production-Grade Computer Vision QA Assistant
            
            This assistant is grounded in carrier SLAs, customs tariffs, delay policies, and **uploaded documents or cargo label images**.
            It uses layout-aware OCR (via Gemini), multimodal embeddings (CLIP), visual taggers, dual FAISS search, and visual grounding verifiers.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat History")
                msg = gr.Textbox(
                    label="Ask a question about the SLA policies, customs rules, or upload an image to query...",
                    placeholder="E.g., What is the tariff code for the items in this invoice?"
                )
                image_input = gr.Image(label="Upload Document Image or Cargo Photo (Optional)", type="filepath")
                username = gr.Textbox(label="Username (for session isolation & memory cache)", value="guest_user")
                clear = gr.ClearButton([msg, image_input, chatbot])
            
            with gr.Column(scale=2):
                gr.Markdown("### 🔍 Live Multimodal Trace")
                trace_json = gr.JSON(label="Metadata & Token Usage")
                sources_md = gr.Markdown(label="Retrieved Source Chunks & Visual Matches")
        
        # When user submits message
        async def user_respond(message, image_path, chat_history, user):
            if chat_history is None:
                chat_history = []
            
            # Formulate user query display text
            user_text = message or ""
            if image_path:
                img_name = Path(image_path).name
                user_text = f"🖼️ [Uploaded: {img_name}] {user_text}".strip()

            bot_msg, trace = await predict(message, image_path, chat_history, user)
            
            chat_history.append({"role": "user", "content": user_text})
            chat_history.append({"role": "assistant", "content": bot_msg})
            
            # Format sources markdown
            sources = trace.get("Retrieved Sources", [])
            sources_text = "#### Retrieved Context:\n"
            if not sources:
                sources_text += "*No references retrieved (direct smalltalk, query safety refusal, or direct image description)*"
            else:
                for idx, src in enumerate(sources):
                    sources_text += f"**Source {idx+1} ({src['Source']})** - Similarity: `{src['Score']:.3f}`\n"
                    sources_text += f"> *{src['Preview']}...*\n\n"
            
            return "", None, chat_history, trace, sources_text
            
        msg.submit(user_respond, [msg, image_input, chatbot, username], [msg, image_input, chatbot, trace_json, sources_md])
        
    return demo


demo = make_ui()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"))
