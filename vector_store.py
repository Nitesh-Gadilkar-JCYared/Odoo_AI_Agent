"""
vector_store.py
───────────────
ChromaDB-based storage for the Odoo AI Assistant.
No Odoo dependency. Data saved to ./chroma_db/ folder.

Stores 3 collections:
  1. query_logs    — every Q&A + generated JSON query + timing + feedback
  2. chat_sessions — session metadata (start time, user, provider)
  3. training_data — wrong/partial answers with corrected queries (for prompt improvement)

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (free, local, ~80MB)
This enables semantic search: "find questions similar to this one"
"""

import json
import uuid
import time
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions


# ── Embedding function (local, free, no API key) ───────────────────────────────
_EMBED_MODEL = "all-MiniLM-L6-v2"

def _get_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=_EMBED_MODEL
    )


class VectorStore:
    """
    Single class that manages all ChromaDB collections.
    Usage:
        store = VectorStore()                      # data saved to ./chroma_db/
        store = VectorStore(path="/my/custom/path")
    """

    def __init__(self, path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        self._emb   = _get_embedding_fn()

        # ── Collections ────────────────────────────────────────────────────────
        # query_logs: one doc per Q&A interaction
        self.query_logs = self.client.get_or_create_collection(
            name               = "query_logs",
            embedding_function = self._emb,
            metadata           = {"hnsw:space": "cosine"},
        )

        # chat_sessions: one doc per session (groups queries)
        self.chat_sessions = self.client.get_or_create_collection(
            name               = "chat_sessions",
            embedding_function = self._emb,
            metadata           = {"hnsw:space": "cosine"},
        )

        # training_data: wrong/partial answers + corrected queries
        self.training_data = self.client.get_or_create_collection(
            name               = "training_data",
            embedding_function = self._emb,
            metadata           = {"hnsw:space": "cosine"},
        )

        print(f"[VectorStore] Connected — path={path}")
        print(f"[VectorStore] query_logs={self.query_logs.count()}  "
              f"sessions={self.chat_sessions.count()}  "
              f"training={self.training_data.count()}")

    # ══════════════════════════════════════════════════════════════════════════
    # SESSIONS
    # ══════════════════════════════════════════════════════════════════════════

    def start_session(self, user: str = "anonymous",
                      llm_provider: str = "groq") -> str:
        """
        Create a new session. Returns session_id (UUID string).
        Call once when the user clicks Connect.
        """
        session_id = str(uuid.uuid4())
        now        = datetime.utcnow().isoformat()

        self.chat_sessions.add(
            ids        = [session_id],
            documents  = [f"Session by {user} using {llm_provider}"],
            metadatas  = [{
                "user":         user,
                "llm_provider": llm_provider,
                "started_at":   now,
                "query_count":  0,
            }],
        )
        print(f"[VectorStore] Session started: {session_id}")
        return session_id

    def update_session_count(self, session_id: str, count: int):
        """Bump query count on the session record."""
        try:
            existing = self.chat_sessions.get(ids=[session_id])
            if existing["ids"]:
                meta = existing["metadatas"][0]
                meta["query_count"] = count
                self.chat_sessions.update(
                    ids       = [session_id],
                    metadatas = [meta],
                )
        except Exception as e:
            print(f"[VectorStore] Warning: session update failed — {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # QUERY LOGS
    # ══════════════════════════════════════════════════════════════════════════

    def log_query(
        self,
        question:        str,
        answer:          str,
        generated_query: dict | str | None = None,
        odoo_model:      str  = "",
        operation:       str  = "unknown",
        record_count:    int  = 0,
        execution_ms:    int  = 0,
        llm_provider:    str  = "groq",
        status:          str  = "success",
        error_message:   str  = "",
        session_id:      str  = "",
    ) -> str:
        """
        Store a complete Q&A log entry.
        The document text = question (what gets embedded for semantic search).
        Returns log_id.
        """
        log_id  = str(uuid.uuid4())
        now     = datetime.utcnow().isoformat()

        # Serialise query dict → JSON string for metadata
        if isinstance(generated_query, dict):
            query_str = json.dumps(generated_query)
        else:
            query_str = generated_query or ""

        metadata = {
            "session_id":    session_id,
            "answer":        answer[:2000],          # ChromaDB metadata limit
            "generated_query": query_str[:1000],
            "odoo_model":    odoo_model,
            "operation":     operation,
            "record_count":  record_count,
            "execution_ms":  execution_ms,
            "llm_provider":  llm_provider,
            "status":        status,
            "error_message": error_message[:500],
            "feedback":      "pending",
            "feedback_note": "",
            "corrected_query": "",
            "in_training":   False,
            "timestamp":     now,
            "date":          now[:10],               # YYYY-MM-DD for date filters
        }

        self.query_logs.add(
            ids       = [log_id],
            documents = [question],                  # ← embedded for semantic search
            metadatas = [metadata],
        )
        print(f"[VectorStore] Logged: {log_id[:8]}… model={odoo_model} status={status}")
        return log_id

    # ══════════════════════════════════════════════════════════════════════════
    # FEEDBACK
    # ══════════════════════════════════════════════════════════════════════════

    def submit_feedback(
        self,
        log_id:    str,
        feedback:  str,            # 'correct' | 'wrong' | 'partial'
        note:      str = "",
        corrected: str = "",       # corrected JSON query for training export
    ) -> bool:
        """
        Update the feedback on an existing log entry.
        If wrong/partial AND corrected query provided → also adds to training_data.
        """
        try:
            existing = self.query_logs.get(ids=[log_id])
            if not existing["ids"]:
                print(f"[VectorStore] Warning: log_id {log_id} not found")
                return False

            meta = existing["metadatas"][0]
            meta["feedback"]       = feedback
            meta["feedback_note"]  = note
            meta["corrected_query"] = corrected

            in_training = feedback in ("wrong", "partial") and bool(corrected)
            meta["in_training"] = in_training

            self.query_logs.update(ids=[log_id], metadatas=[meta])

            # Also write to training_data collection if corrected
            if in_training:
                self._add_to_training(
                    log_id   = log_id,
                    question = existing["documents"][0],
                    wrong    = meta.get("generated_query", ""),
                    correct  = corrected,
                    feedback = feedback,
                    note     = note,
                    model    = meta.get("odoo_model", ""),
                )

            print(f"[VectorStore] Feedback saved: {log_id[:8]}… → {feedback}")
            return True

        except Exception as e:
            print(f"[VectorStore] Warning: feedback save failed — {e}")
            return False

    def _add_to_training(self, log_id, question, wrong, correct,
                         feedback, note, model):
        """Add a corrected example to the training_data collection."""
        training_doc = (
            f"QUESTION: {question}\n"
            f"WRONG QUERY: {wrong}\n"
            f"CORRECT QUERY: {correct}"
        )
        self.training_data.add(
            ids       = [f"train_{log_id}"],
            documents = [training_doc],
            metadatas = [{
                "question":       question,
                "wrong_query":    wrong,
                "correct_query":  correct,
                "feedback":       feedback,
                "feedback_note":  note,
                "odoo_model":     model,
                "timestamp":      datetime.utcnow().isoformat(),
            }],
        )
        print(f"[VectorStore] Added to training_data: {log_id[:8]}…")

    # ══════════════════════════════════════════════════════════════════════════
    # SEMANTIC SEARCH
    # ══════════════════════════════════════════════════════════════════════════

    def find_similar_questions(
        self,
        question: str,
        n:        int = 5,
        only_correct: bool = True,
    ) -> list[dict]:
        """
        Find semantically similar past questions using embeddings.
        Use this to:
          - Show users "similar questions others asked"
          - Find correct past answers to reuse (RAG-style)
          - Spot duplicate/redundant questions

        Returns list of dicts: {question, answer, feedback, similarity}
        """
        where = {"feedback": "correct"} if only_correct else None

        try:
            results = self.query_logs.query(
                query_texts = [question],
                n_results   = min(n, max(self.query_logs.count(), 1)),
                where       = where,
            )

            hits = []
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i]
                hits.append({
                    "question":   doc,
                    "answer":     meta.get("answer", ""),
                    "feedback":   meta.get("feedback", "pending"),
                    "model":      meta.get("odoo_model", ""),
                    "similarity": round(1 - dist, 3),   # cosine → similarity
                    "log_id":     results["ids"][0][i],
                })
            return hits

        except Exception as e:
            print(f"[VectorStore] Semantic search failed — {e}")
            return []

    def find_similar_training(self, question: str, n: int = 3) -> list[dict]:
        """
        Find similar examples from the training_data collection.
        Use this to inject few-shot examples into prompts dynamically.
        Returns list of {question, correct_query, similarity}
        """
        if self.training_data.count() == 0:
            return []
        try:
            results = self.training_data.query(
                query_texts = [question],
                n_results   = min(n, self.training_data.count()),
            )
            hits = []
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i]
                hits.append({
                    "question":      meta.get("question", ""),
                    "correct_query": meta.get("correct_query", ""),
                    "similarity":    round(1 - dist, 3),
                })
            return hits
        except Exception as e:
            print(f"[VectorStore] Training search failed — {e}")
            return []

    # ══════════════════════════════════════════════════════════════════════════
    # STATS & RETRIEVAL
    # ══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        """Overall accuracy stats for the sidebar."""
        try:
            total = self.query_logs.count()
            if total == 0:
                return {"total": 0, "correct": 0, "wrong": 0,
                        "partial": 0, "errors": 0, "accuracy": 0.0}

            correct = len(self.query_logs.get(
                where={"feedback": "correct"})["ids"])
            wrong   = len(self.query_logs.get(
                where={"feedback": "wrong"})["ids"])
            partial = len(self.query_logs.get(
                where={"feedback": "partial"})["ids"])
            errors  = len(self.query_logs.get(
                where={"status": "error"})["ids"])
            rated   = correct + wrong + partial
            accuracy = round(correct / rated * 100, 1) if rated > 0 else 0.0

            return {
                "total":    total,
                "correct":  correct,
                "wrong":    wrong,
                "partial":  partial,
                "errors":   errors,
                "accuracy": accuracy,
                "training": self.training_data.count(),
            }
        except Exception as e:
            print(f"[VectorStore] Stats failed — {e}")
            return {}

    def get_recent_logs(self, n: int = 20) -> list[dict]:
        """Return most recent n query logs (for history view)."""
        try:
            results = self.query_logs.get(
                limit=n,
                include=["documents", "metadatas"],
            )
            logs = []
            for i, doc in enumerate(results["documents"]):
                meta = results["metadatas"][i]
                logs.append({
                    "log_id":   results["ids"][i],
                    "question": doc,
                    **meta,
                })
            # Sort by timestamp descending
            logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return logs
        except Exception as e:
            print(f"[VectorStore] get_recent_logs failed — {e}")
            return []

    def export_training_jsonl(self) -> str:
        """
        Export all training_data as JSONL string.
        Each line: {"question": ..., "wrong_query": ..., "correct_query": ...}
        Use to build few-shot prompt examples.
        """
        try:
            results = self.training_data.get(
                include=["metadatas"]
            )
            lines = []
            for meta in results["metadatas"]:
                lines.append(json.dumps({
                    "question":      meta.get("question", ""),
                    "wrong_query":   meta.get("wrong_query", ""),
                    "correct_query": meta.get("correct_query", ""),
                    "odoo_model":    meta.get("odoo_model", ""),
                    "feedback_note": meta.get("feedback_note", ""),
                }, ensure_ascii=False))
            return "\n".join(lines)
        except Exception as e:
            print(f"[VectorStore] Export failed — {e}")
            return ""
