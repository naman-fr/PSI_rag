import os
import sys
import asyncio
import gradio as gr
from pathlib import Path

# Satisfy Pydantic Settings validation if keys are missing from environment
if not os.environ.get("GROQ_API_KEY") and not os.path.exists(".env"):
    os.environ["GROQ_API_KEY"] = "missing-key-placeholder"
if not os.environ.get("GEMINI_API_KEY") and not os.path.exists(".env"):
    os.environ["GEMINI_API_KEY"] = "missing-key-placeholder"

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
    logger.info("Initializing RAG pipeline for Hugging Face Space...")

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
    # Ensure event loop runs cleanly
    asyncio.run(init_rag())
except Exception as e:
    import traceback
    init_error = f"Startup Event Loop Exception: {str(e)}\n{traceback.format_exc()}"
    logger.exception("Error running RAG initialization on startup", error=str(e))


async def predict(message, history, username):
    global orchestrator, init_error
    if not orchestrator:
        err_msg = f"System is not initialized. Please verify your environment and API keys.\n\nInitialization Error:\n{init_error or 'Unknown Error'}"
        return err_msg, {}

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
            "Vector Backend": orchestrator.retriever._backend if (orchestrator and hasattr(orchestrator, "retriever")) else "unknown",
            "Total Chunks": orchestrator.retriever.count if (orchestrator and hasattr(orchestrator, "retriever")) else 0,
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
        import traceback
        tb = traceback.format_exc()
        logger.exception("Error processing chat message", error=str(e))
        return f"An error occurred: {str(e)}\n\nTraceback:\n{tb}", {"error": str(e), "traceback": tb}


def make_ui():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # 🚢 PSI RAG: Production Guardrailed Self-RAG
            ### GlobalFreight Logistics Document QA Assistant
            """
        )
        
        with gr.Tab("💬 Chat Interface"):
            gr.Markdown(
                """
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

        with gr.Tab("📊 Ragas Evaluation Dashboard"):
            gr.Markdown(
                """
                ### Ragas RAG Pipeline Performance Metrics
                Evaluate the factual accuracy and quality of generated responses using Google Gemini as the judge.
                
                - **Faithfulness**: Measures if the generated response is factually consistent and fully supported by the retrieved contexts.
                - **Answer Relevancy**: Measures how pertinent the generated response is to the user's initial question.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    run_btn = gr.Button("🚀 Run Ragas Evaluation (3 Test Cases)", variant="primary")
                    status_txt = gr.Textbox(label="Status", value="Ready", interactive=False)
                    
                with gr.Column(scale=2):
                    summary_md = gr.Markdown("### Aggregate Scores\n*No evaluation data loaded.*")
            
            df_table = gr.Dataframe(
                headers=["user_input", "response", "faithfulness", "answer_relevancy", "context_precision"],
                datatype=["str", "str", "number", "number", "number"],
                label="Detailed Ragas Score Breakdown"
            )
            
            # Helper function to load latest results
            def load_latest():
                import pandas as pd
                import json
                results_path = Path("data/ragas_eval_results.json")
                if results_path.exists():
                    try:
                        with open(results_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if data:
                            df = pd.DataFrame(data)
                            f_avg = df["faithfulness"].mean() if "faithfulness" in df.columns else 0.0
                            r_avg = df["answer_relevancy"].mean() if "answer_relevancy" in df.columns else 0.0
                            c_avg = df["context_precision"].mean() if "context_precision" in df.columns else 0.0
                            summary = f"### Aggregate Scores\n- **Average Faithfulness:** `{f_avg:.4f}`\n- **Average Answer Relevancy:** `{r_avg:.4f}`\n- **Average Context Precision:** `{c_avg:.4f}`"
                            
                            # Clean up df columns for table representation if they exist
                            cols_to_keep = ["user_input", "response", "faithfulness", "answer_relevancy", "context_precision"]
                            df_clean = df[[c for c in cols_to_keep if c in df.columns]]
                            return df_clean, summary, "Ready (Loaded last saved results)"
                    except Exception as e:
                        logger.warning("Failed to load saved Ragas evaluation results: %s", str(e))
                return pd.DataFrame(), "### Aggregate Scores\n*No evaluation data loaded. Click Run to evaluate.*", "Ready"
            
            # Action when Run button is clicked
            async def trigger_evaluation():
                import pandas as pd
                from scripts.run_ragas_eval import run_ragas_eval
                from app.core.config import get_settings
                
                settings = get_settings()
                
                # Check if we should use mock mode
                is_mock = False
                # If keys are placeholders or not set, fall back to mock mode
                if (settings.gemini_api_key in ["mock-gemini-key", "missing-key-placeholder", ""] or 
                    settings.groq_api_key in ["mock-groq-key", "missing-key-placeholder", ""]):
                    is_mock = True
                
                eval_questions = [
                    "What is the delivery commitment for Platinum Express?",
                    "What compensation is given for delayed Platinum shipments?",
                    "What is the delay tolerance for Gold Standard?"
                ]
                
                status_msg = "Running Ragas evaluation in " + ("MOCK" if is_mock else "REAL") + " mode..."
                logger.info(status_msg)
                
                try:
                    df, f_avg, r_avg, c_avg = await run_ragas_eval(
                        mock_mode=is_mock,
                        questions_list=eval_questions,
                        orchestrator_instance=orchestrator
                    )
                    
                    if df is None:
                        return pd.DataFrame(), "### Aggregate Scores\n*No data was collected during evaluation.*", "Error: No RAG data collected"
                    
                    summary = f"### Aggregate Scores ({'Mock Mode' if is_mock else 'Real Mode'})\n- **Average Faithfulness:** `{f_avg:.4f}`\n- **Average Answer Relevancy:** `{r_avg:.4f}`\n- **Average Context Precision:** `{c_avg:.4f}`"
                    
                    cols_to_keep = ["user_input", "response", "faithfulness", "answer_relevancy", "context_precision"]
                    df_clean = df[[c for c in cols_to_keep if c in df.columns]]
                    
                    final_status = "Evaluation completed successfully (" + ("Mock" if is_mock else "Real") + " Mode)"
                    return df_clean, summary, final_status
                except Exception as e:
                    logger.exception("Evaluation action failed", error=str(e))
                    return pd.DataFrame(), f"### Aggregate Scores\n*Evaluation failed: {str(e)}*", "Error during evaluation"

            # Set load handler to load saved results
            demo.load(load_latest, None, [df_table, summary_md, status_txt])
            
            # Bind click event
            run_btn.click(
                trigger_evaluation,
                inputs=None,
                outputs=[df_table, summary_md, status_txt]
            )

    return demo


demo = make_ui()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"))
