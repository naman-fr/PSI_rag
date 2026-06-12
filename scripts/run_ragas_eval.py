#!/usr/bin/env python
"""Ragas Evaluation script for evaluating the PSI RAG pipeline.

Supports faithfulness and answer relevancy metrics using Google Gemini.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

# If running in mock mode or if keys are missing from environment, satisfy Pydantic Settings validation
# so that we can check and print a clean error message inside main.
keys_are_missing = False
if not os.environ.get("GROQ_API_KEY") and not os.path.exists(".env"):
    os.environ["GROQ_API_KEY"] = "missing-key-placeholder"
    keys_are_missing = True
if not os.environ.get("GEMINI_API_KEY") and not os.path.exists(".env"):
    os.environ["GEMINI_API_KEY"] = "missing-key-placeholder"
    keys_are_missing = True

if "--mock" in sys.argv:
    # Overwrite placeholders with mock keys for mock execution
    os.environ["GROQ_API_KEY"] = "mock-groq-key"
    os.environ["GEMINI_API_KEY"] = "mock-gemini-key"
    os.environ.setdefault("PINECONE_API_KEY", "mock-pinecone-key")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

from dotenv import load_dotenv
from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger

load_dotenv()
setup_logging()
logger = get_logger("ragas_evaluation")


class MockEvaluationResult:
    """Mock Ragas EvaluationResult for testing plumbing."""
    def __init__(self, scores, df):
        self.scores = scores
        self._df = df

    def to_pandas(self):
        return self._df

    def __str__(self):
        return str(self.scores)

    def __repr__(self):
        return repr(self.scores)

    def __getitem__(self, key):
        return self.scores[key]


from ragas.llms.base import LangchainLLMWrapper

class GroqLangchainLLMWrapper(LangchainLLMWrapper):
    """Custom wrapper for LangChain ChatGroq to handle n > 1 constraint by executing n=1 requests in parallel."""
    
    def generate_text(
        self,
        prompt,
        n: int = 1,
        temperature = None,
        stop = None,
        callbacks = None,
    ):
        if n <= 1:
            return super().generate_text(prompt, n, temperature, stop, callbacks)
            
        logger.info(f"GroqLangchainLLMWrapper: intercepting generate_text for n={n} and running sequentially")
        results = [
            super(GroqLangchainLLMWrapper, self).generate_text(
                prompt, n=1, temperature=temperature, stop=stop, callbacks=callbacks
            )
            for _ in range(n)
        ]
        
        combined_generations = []
        for res in results:
            if res.generations and len(res.generations) > 0:
                combined_generations.extend(res.generations[0])
                
        from langchain_core.outputs import LLMResult
        return LLMResult(
            generations=[combined_generations],
            llm_output=results[0].llm_output if results else None,
            run=results[0].run if results else None
        )

    async def agenerate_text(
        self,
        prompt,
        n: int = 1,
        temperature = None,
        stop = None,
        callbacks = None,
    ):
        if n <= 1:
            return await super().agenerate_text(prompt, n, temperature, stop, callbacks)
            
        logger.info(f"GroqLangchainLLMWrapper: intercepting agenerate_text for n={n} and running in parallel")
        tasks = [
            super(GroqLangchainLLMWrapper, self).agenerate_text(
                prompt, n=1, temperature=temperature, stop=stop, callbacks=callbacks
            )
            for _ in range(n)
        ]
        results = await asyncio.gather(*tasks)
        
        combined_generations = []
        for res in results:
            if res.generations and len(res.generations) > 0:
                combined_generations.extend(res.generations[0])
                
        from langchain_core.outputs import LLMResult
        return LLMResult(
            generations=[combined_generations],
            llm_output=results[0].llm_output if results else None,
            run=results[0].run if results else None
        )
from ragas.embeddings.base import BaseRagasEmbeddings

class RagasEmbeddingWrapper(BaseRagasEmbeddings):
    """Custom wrapper for Ragas to use our working EmbeddingService."""
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        super().__init__()
        
    def embed_query(self, text: str) -> list[float]:
        vec = self.embedding_service.embed_query(text)
        return [float(x) for x in vec]
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.embedding_service.embed_documents(texts)
        return [[float(x) for x in v] for v in vecs]


def _safe_float(val):
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        if isinstance(val, list):
            clean_vals = []
            for v in val:
                try:
                    if v is not None:
                        clean_vals.append(float(v))
                except (TypeError, ValueError):
                    pass
            return sum(clean_vals) / len(clean_vals) if clean_vals else 0.0
        return 0.0


async def run_ragas_eval(num_questions: str = "5", mock_mode: bool = False, questions_list: list = None):
    settings = get_settings()

    if questions_list:
        questions = questions_list
    else:
        # Load test questions
        questions_path = Path(__file__).parent.parent / "tests" / "fixtures" / "test_questions.json"
        with open(questions_path, "r", encoding="utf-8") as f:
            all_questions = json.load(f)

        # Filter/Slice questions
        if num_questions.lower() == "all":
            questions = all_questions
        else:
            try:
                limit = int(num_questions)
                questions = all_questions[:limit]
            except ValueError:
                print(f"Invalid value for num-questions: {num_questions}. Using default of 5.")
                questions = all_questions[:5]

    logger.info("Loaded questions for evaluation", count=len(questions), mock_mode=mock_mode)

    # Initialize local RAG components
    from app.cache.redis_client import InMemoryCache
    from app.memory.conversation import ConversationManager
    from app.memory.summary import SummaryManager
    from app.rag.retrieval import FAISSRetriever
    from app.services.orchestrator import RAGOrchestrator

    cache = InMemoryCache()
    conv_mgr = ConversationManager(cache)
    sum_mgr = SummaryManager(cache)

    if mock_mode:
        from unittest.mock import AsyncMock, MagicMock
        import numpy as np

        # Mock embedding service
        embedding_service = MagicMock()
        mock_vec = np.zeros(settings.embed_dimension, dtype=np.float32)
        embedding_service.embed_query.return_value = mock_vec
        embedding_service.embed_documents.return_value = [mock_vec]

        # Mock retriever
        retriever = MagicMock(spec=FAISSRetriever)
        retriever.search.return_value = [
            {
                "score": 0.95,
                "text": "Gold tier delay tolerance is 12 hours. Compensation is 10% refund of freight charges for each 24-hour block.",
                "metadata": {"source": "DOC1_carrier_sla_agreement.md"},
            },
            {
                "score": 0.88,
                "text": "Platinum Express delay tolerance is 4 hours. Compensation is 15% refund of freight charges for each 24-hour block.",
                "metadata": {"source": "DOC1_carrier_sla_agreement.md"},
            },
            {
                "score": 0.85,
                "text": "Category 2 significant delay is 4 to 24 hours. Action: update ETA, notify customer, agent proactive outreach within 1 hour.",
                "metadata": {"source": "DOC3_shipment_delay_policy.md"},
            }
        ]

        # Mock LLM
        llm_service = AsyncMock()
        llm_service.generate.return_value = (
            "Based on the provided context, the delay tolerance for Gold is 12 hours (10% refund per 24h) and Platinum is 4 hours (15% refund per 24h). Under 15 hours delay (Category 2 significant delay), we must updates ETA, notify customer, and perform proactive agent outreach for Platinum within 1 hour.",
            {"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180},
        )
    else:
        # Check API Keys
        if keys_are_missing or settings.gemini_api_key == "missing-key-placeholder" or settings.groq_api_key == "missing-key-placeholder":
            print("\n" + "=" * 80)
            print("ERROR: REQUIRED API KEYS ARE MISSING")
            print("=" * 80)
            print("To run the real Ragas evaluation, you need to configure your API keys.")
            print("Please define GROQ_API_KEY and GEMINI_API_KEY in your environment, or")
            print("create a '.env' file in the project root with the following format:")
            print("\nGROQ_API_KEY=your-groq-api-key")
            print("GEMINI_API_KEY=your-google-gemini-api-key")
            print("\nAlternatively, you can test the script plumbing in mock mode by running:")
            print("python scripts/run_ragas_eval.py --mock\n")
            sys.exit(1)

        from app.rag.embeddings import EmbeddingService
        from app.rag.generation import LLMService

        embedding_service = EmbeddingService()
        retriever = FAISSRetriever(dimension=settings.embed_dimension)

        # Load FAISS index
        index_path = f"{settings.index_dir}/faiss.index"
        if not Path(index_path).exists():
            logger.info("FAISS index not found. Running ingestion first...")
            from app.main import run_ingestion
            await run_ingestion(source_dir=settings.docs_dir)
        else:
            await retriever.load_index(index_path)

        llm_service = LLMService()

    orchestrator = RAGOrchestrator(
        embedding_service=embedding_service,
        retriever=retriever,
        llm_service=llm_service,
        cache_backend=cache,
        conversation_manager=conv_mgr,
        summary_manager=sum_mgr,
    )

    rag_samples = []
    print("\n" + "=" * 80)
    print(f"RUNNING PIPELINE TO COLLECT EVALUATION DATA ({'MOCK' if mock_mode else 'REAL'})")
    print("=" * 80)

    for i, q in enumerate(questions):
        t0 = time.time()
        try:
            response = await orchestrator.process_query(
                question=q,
                username=f"ragas_user_{i}",
                session_id=f"ragas_sess_{i}",
            )
            elapsed = time.time() - t0

            # Only evaluate RAG retrieval queries
            is_rag = response.get("mode") == "retrieval"
            retrieved = response.get("retrieved_contexts", [])

            print(f"[{i+1}/{len(questions)}] Q: {q}")
            print(f"       -> Mode: {response.get('mode')}, Contexts: {len(retrieved)}, Time: {elapsed:.2f}s")

            if is_rag and retrieved:
                rag_samples.append({
                    "user_input": q,
                    "response": response.get("answer", ""),
                    "retrieved_contexts": retrieved
                })
        except Exception as e:
            logger.exception("Error processing query", question=q, error=str(e))

    if not rag_samples:
        print("\nNo RAG retrieval queries collected. Check that your queries trigger 'retrieval' mode.")
        return None, 0.0, 0.0, 0.0

    print(f"\nCollected {len(rag_samples)} samples for Ragas evaluation.")

    print("\n" + "=" * 80)
    print("RUNNING RAGAS METRICS EVALUATION")
    print("=" * 80)

    if mock_mode:
        # Create mock DataFrame and EvaluationResult
        import pandas as pd
        mock_df = pd.DataFrame(rag_samples)
        mock_df["faithfulness"] = [0.92 - (idx * 0.02) for idx in range(len(rag_samples))]
        mock_df["answer_relevancy"] = [0.95 - (idx * 0.01) for idx in range(len(rag_samples))]
        mock_df["context_precision"] = [0.88 + (idx * 0.03) for idx in range(len(rag_samples))]

        result = MockEvaluationResult(
            scores={
                "faithfulness": mock_df["faithfulness"].mean(),
                "answer_relevancy": mock_df["answer_relevancy"].mean(),
                "context_precision": mock_df["context_precision"].mean(),
            },
            df=mock_df
        )
        # Simulate slight delay
        await asyncio.sleep(1)
    else:
        # Load real Ragas evaluation
        from google import genai
        from ragas.llms import llm_factory
        from ragas.embeddings.base import embedding_factory
        from ragas import SingleTurnSample, EvaluationDataset, aevaluate
        from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithoutReference

        # Initialize Google GenAI client for Ragas judge
        client = genai.Client(api_key=settings.gemini_api_key)
        
        # Check if Gemini LLM is working (has remaining quota)
        use_groq_fallback = False
        try:
            logger.info("Testing Gemini LLM availability for Ragas evaluation...")
            client.models.generate_content(
                model="gemini-2.0-flash",
                contents="Hello",
            )
            logger.info("Gemini LLM test call succeeded.")
        except Exception as e:
            logger.warning("Gemini LLM test call failed. Falling back to Groq (llama-3.3-70b-versatile).", error=str(e))
            use_groq_fallback = True

        if use_groq_fallback:
            groq_key = os.environ.get("GROQ_API_KEY") or settings.groq_api_key
            if groq_key and groq_key != "missing-key-placeholder":
                logger.info("Initializing Ragas judge LLM using Groq (llama-3.3-70b-versatile)...")
                from langchain_groq import ChatGroq
                chat_model = ChatGroq(
                    api_key=groq_key,
                    model="llama-3.3-70b-versatile",
                    temperature=0
                )
                ragas_llm = GroqLangchainLLMWrapper(chat_model)
            else:
                logger.warning("Groq API key is not configured; cannot fall back. Attempting with Gemini anyway.")
                ragas_llm = llm_factory(
                    model="gemini-2.0-flash",
                    provider="google",
                    client=client
                )
        else:
            ragas_llm = llm_factory(
                model="gemini-2.0-flash",
                provider="google",
                client=client
            )

        # Wrap our own working embedding service
        ragas_embeddings = RagasEmbeddingWrapper(embedding_service)

        # Instantiate metrics with proper dependencies
        faithfulness_metric = Faithfulness(llm=ragas_llm)
        answer_relevancy_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
        context_precision_metric = LLMContextPrecisionWithoutReference(llm=ragas_llm)

        # Convert to Ragas SingleTurnSamples
        samples = []
        for s in rag_samples:
            samples.append(
                SingleTurnSample(
                    user_input=s["user_input"],
                    response=s["response"],
                    retrieved_contexts=s["retrieved_contexts"]
                )
            )

        dataset = EvaluationDataset(samples=samples)

        from ragas.run_config import RunConfig

        run_config = RunConfig(
            max_workers=1,
            timeout=300,
            max_retries=10,
            max_wait=60
        )

        # Run Ragas evaluate
        result = await aevaluate(
            dataset=dataset,
            metrics=[faithfulness_metric, answer_relevancy_metric, context_precision_metric],
            run_config=run_config
        )

    # Display results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    
    try:
        f_val = result['faithfulness']
    except Exception:
        f_val = None
        
    try:
        r_val = result['answer_relevancy']
    except Exception:
        r_val = None

    try:
        c_val = result['context_precision']
    except Exception:
        c_val = None

    f_avg = _safe_float(f_val)
    r_avg = _safe_float(r_val)
    c_avg = _safe_float(c_val)

    print(f"Average Faithfulness Score:  {f_avg:.4f}")
    print(f"Average Answer Relevancy:    {r_avg:.4f}")
    print(f"Average Context Precision:   {c_avg:.4f}")
    print("-" * 80)

    df = result.to_pandas()
    print("\nDetailed breakdown:")
    for idx, row in df.iterrows():
        print(f"\nQ: {row['user_input']}")
        print(f"A: {row['response'][:100]}...")
        f_row = _safe_float(row.get('faithfulness')) if hasattr(row, 'get') else _safe_float(row['faithfulness'])
        r_row = _safe_float(row.get('answer_relevancy')) if hasattr(row, 'get') else _safe_float(row['answer_relevancy'])
        c_row = _safe_float(row.get('context_precision')) if hasattr(row, 'get') else _safe_float(row['context_precision'])
        print(f"-> Faithfulness: {f_row:.2f} | Relevancy: {r_row:.2f} | Context Precision: {c_row:.2f}")

    # Save to disk
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    records = df.to_dict(orient="records")
    with open(output_dir / "ragas_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nResults successfully saved to data/ragas_eval_results.json")
    return df, f_avg, r_avg, c_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ragas Evaluation on RAG pipeline.")
    parser.add_argument(
        "--num-questions",
        type=str,
        default="5",
        help="Number of questions to evaluate (e.g. 5, 10, or 'all')."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode without invoking LLM endpoints."
    )
    args = parser.parse_args()

    asyncio.run(run_ragas_eval(args.num-questions if hasattr(args, "num-questions") else args.num_questions, mock_mode=args.mock))
