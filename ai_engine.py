"""
ai_engine.py
────────────
Step 1: Detect intent — fields question OR data question
Step 2: For fields questions — ask Odoo which model matches, return fields table
Step 3: For data questions — build live-schema prompt, LLM → JSON query → execute
Step 4: Format answer, log to ChromaDB
Step 5: Inject similar past correct queries as dynamic few-shot examples
"""

import json
import re
import time
from datetime import datetime

from odoo_connector import OdooConnector
from llm_client import GroqClient


# ── Prompt: resolve natural language → Odoo model name ────────────────────────
MODEL_RESOLVER_PROMPT = """You are an Odoo ERP expert. Given a user question, return ONLY
the Odoo technical model name (e.g. sale.order, account.move, stock.quant).

If the question mentions multiple models, return the most relevant one.
If you are not sure, return your best guess — always return something.

Return ONLY the model name. No explanation. No punctuation. Just the model name.

Examples:
Q: what fields does the sales order table have?      → sale.order
Q: show me fields in invoices                        → account.move
Q: what columns are in stock moves                   → stock.move
Q: fields of purchase order lines                    → purchase.order.line
Q: what fields does res partner have                 → res.partner
Q: show columns in hr leave                          → hr.leave
Q: fields in account journal                         → account.journal
Q: what fields does mrp bom have                     → mrp.bom
Q: columns in project task                           → project.task
Q: fields of sale order line                         → sale.order.line
Q: what columns in account payment                   → account.payment
Q: stock location fields                             → stock.location
Q: fields of product category                        → product.category"""


# ── Prompt: resolve question → Odoo model for data queries ────────────────────
QUERY_MODEL_RESOLVER_PROMPT = """You are an Odoo ERP expert. Given a user question about
business data, return ONLY the most relevant Odoo technical model name.

Return ONLY the model name string. Nothing else.

Examples:
Q: show me unpaid invoices                     → account.move
Q: top customers by revenue                    → sale.order
Q: which products are low on stock             → stock.quant
Q: open purchase orders                        → purchase.order
Q: list all employees in IT department         → hr.employee
Q: show CRM leads with high probability        → crm.lead
Q: pending delivery orders                     → stock.picking
Q: vendor bills due this month                 → account.move
Q: sales order lines for customer ACME         → sale.order.line
Q: account journal entries                     → account.move.line
Q: time off requests pending approval          → hr.leave
Q: manufacturing orders in progress            → mrp.production"""


# ── Main query generation prompt ───────────────────────────────────────────────
QUERY_PROMPT_TEMPLATE = """You generate Odoo ERP JSON-RPC queries from natural language.

TODAY: {today}
MONTH START: {month_start}
YEAR START: {year_start}

━━━ FIELDS OF THE TARGET MODEL (live from Odoo) ━━━
{model_schema}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPERATIONS:
1. search_read  — list records with filters
2. read_group   — totals, aggregations, group by
3. search_count — count only

DOMAIN OPERATORS: = != > < >= <= like ilike in not in
DATE FORMAT: "YYYY-MM-DD"

FIELD RULES:
- Only use field names shown in the schema above
- many2one filter: ["partner_id.name", "ilike", "text"]
- aggregation in read_group: "amount_total:sum"
- selection values: use exactly the values listed after "selection:"

OUTPUT: return ONLY raw JSON. No markdown, no explanation, no backticks.

search_read  → {{"operation":"search_read","model":"{model}","domain":[...],"fields":["f1","f2"],"order":"field desc","limit":20}}
read_group   → {{"operation":"read_group","model":"{model}","domain":[...],"fields":["field","amount:sum"],"groupby":["field"],"order":"amount desc","limit":10}}
search_count → {{"operation":"search_count","model":"{model}","domain":[...]}}

EXAMPLES:
Q: top 5 customers by sales this month
A: {{"operation":"read_group","model":"sale.order","domain":[["state","=","sale"],["date_order",">=","{month_start}"]],"fields":["partner_id","amount_total:sum"],"groupby":["partner_id"],"order":"amount_total desc","limit":5}}

Q: unpaid customer invoices
A: {{"operation":"search_read","model":"account.move","domain":[["move_type","=","out_invoice"],["payment_state","=","not_paid"],["state","=","posted"]],"fields":["name","partner_id","amount_total","amount_residual","invoice_date_due"],"order":"invoice_date_due asc","limit":50}}

{dynamic_examples}

Output must start with {{ and end with }}, nothing else."""


FORMAT_PROMPT = """You are a business data analyst. Present Odoo ERP data clearly.
- Use markdown tables for lists of records
- Format currency with 2 decimal places and commas
- For fields like [42, "Customer Name"], show only the name
- Give 1-2 key insights at the end
- Be concise and clear"""


FIELDS_QUESTION_PATTERNS = [
    "what fields", "which fields", "list fields", "show fields",
    "all fields", "fields of", "fields in", "fields does",
    "columns of", "columns in", "what columns", "which columns",
    "schema of", "structure of", "show me fields", "get fields",
]


class OdooAIEngine:

    def __init__(self, connector: OdooConnector, groq_api_key: str,
                 vector_store=None):
        self.connector    = connector
        self.llm          = GroqClient(groq_api_key)
        self.vs           = vector_store
        self.session_id   = None
        self._query_count = 0

    def set_session(self, session_id: str):
        self.session_id = session_id

    def clear_history(self):
        self._query_count = 0

    def refresh_schema(self):
        self.connector.clear_fields_cache()
        print("[ENGINE] Field cache cleared — will re-fetch on next question.")

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL RESOLVER — LLM figures out which model the user means
    # ══════════════════════════════════════════════════════════════════════════

    def _resolve_model(self, question: str, for_fields: bool = False) -> str:
        """
        Use the LLM to convert natural language → Odoo model name.
        Works for ANY model installed in Odoo — not limited to a whitelist.
        """
        prompt = MODEL_RESOLVER_PROMPT if for_fields else QUERY_MODEL_RESOLVER_PROMPT
        raw = self.llm.chat(
            system       = prompt,
            user_message = question,
            max_tokens   = 30,
        ).strip().lower()

        # Clean up — remove quotes, extra spaces, trailing punctuation
        model = re.sub(r"[\"'\s]", "", raw).strip(".")
        print(f"[ENGINE] Resolved model: '{model}' for question: '{question[:60]}'")
        return model

    # ══════════════════════════════════════════════════════════════════════════
    # FIELDS INTROSPECTION — answers "what fields does X have?"
    # ══════════════════════════════════════════════════════════════════════════

    def get_model_fields_answer(self, question: str) -> str:
        """
        Handle any 'what fields does X model have?' question.
        Uses LLM to identify model, then calls fields_get() on Odoo directly.
        No whitelist — works for any installed model.
        """
        # Step 1: Ask LLM which model the user means
        model = self._resolve_model(question, for_fields=True)

        if not model or "." not in model:
            # Couldn't resolve — list all available models
            try:
                all_models = self.connector.list_all_models()
                lines = [
                    "I couldn't identify which model you meant. "
                    "Here are all installed models:\n",
                    "| Model | Name |",
                    "|---|---|",
                ]
                for m in all_models[:80]:
                    lines.append(f"| `{m['model']}` | {m['name']} |")
                if len(all_models) > 80:
                    lines.append(f"\n_...and {len(all_models)-80} more. "
                                 f"Ask about a specific one._")
                return "\n".join(lines)
            except Exception as e:
                return f"⚠️ Could not list models: {e}"

        # Step 2: Verify model exists in this Odoo instance
        if not self.connector.model_exists(model):
            # Try to suggest similar model names
            try:
                keyword   = model.split(".")[0]
                similar   = self.connector.list_all_models(keyword=keyword)
                sug_lines = [f"⚠️ Model `{model}` not found in this Odoo instance.\n"]
                if similar:
                    sug_lines.append("Did you mean one of these?\n")
                    for m in similar[:10]:
                        sug_lines.append(f"- `{m['model']}` — {m['name']}")
                return "\n".join(sug_lines)
            except Exception:
                return f"⚠️ Model `{model}` not found in this Odoo instance."

        # Step 3: Fetch and format fields
        try:
            fields = self.connector.get_fields(model, filter_useful=True)

            if not fields:
                return f"⚠️ No queryable fields found for `{model}`."

            lines = [
                f"### Fields of `{model}`  ({len(fields)} fields)\n",
                "| Field Name | Type | Label | Required |",
                "|---|---|---|---|",
            ]
            for fname, meta in sorted(fields.items()):
                ftype    = meta.get("type", "?")
                label    = meta.get("string", fname)
                req      = "★" if meta.get("required") else ""
                relation = meta.get("relation", "")
                choices  = meta.get("selection", [])

                if ftype == "many2one" and relation:
                    type_display = f"`many2one` → `{relation}`"
                elif ftype in ("one2many", "many2many") and relation:
                    type_display = f"`{ftype}` → `{relation}`"
                elif ftype == "selection" and choices:
                    vals         = " \\| ".join(str(v) for v, _ in choices)
                    type_display = f"`selection`: {vals}"
                else:
                    type_display = f"`{ftype}`"

                lines.append(f"| `{fname}` | {type_display} | {label} | {req} |")

            first_field = sorted(fields.keys())[0]
            lines.append(
                f"\n💡 Use these field names in your questions — "
                f"e.g. *'show {model} sorted by {first_field}'*"
            )
            return "\n".join(lines)

        except Exception as e:
            return f"⚠️ Error fetching fields for `{model}`: {e}"

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN ASK
    # ══════════════════════════════════════════════════════════════════════════

    def ask(self, question: str, llm_provider: str = "groq") -> dict:
        t_start     = time.time()
        today       = datetime.now().strftime("%Y-%m-%d")
        month_start = datetime.now().strftime("%Y-%m-01")
        year_start  = datetime.now().strftime("%Y-01-01")

        # ── Route: fields question? ────────────────────────────────────────────
        q_lower = question.lower()
        if any(pat in q_lower for pat in FIELDS_QUESTION_PATTERNS):
            answer = self.get_model_fields_answer(question)
            log_id = None
            if self.vs:
                log_id = self.vs.log_query(
                    question     = question,
                    answer       = answer,
                    odoo_model   = "fields_get",
                    operation    = "fields_get",
                    status       = "success",
                    session_id   = self.session_id or "",
                    execution_ms = int((time.time() - t_start) * 1000),
                )
            return {
                "answer": answer, "model": "fields_get", "record_count": 0,
                "log_id": log_id, "query": None, "status": "success",
                "execution_ms": int((time.time() - t_start) * 1000),
            }

        # ── Route: data question ───────────────────────────────────────────────

        # Step 1: Resolve which model to query
        target_model = self._resolve_model(question, for_fields=False)

        # Step 2: Fetch live field schema for that model (cached after first call)
        model_schema = ""
        if target_model and "." in target_model:
            try:
                model_schema = self.connector.get_fields_summary(target_model)
            except Exception as e:
                print(f"[ENGINE] Warning: could not fetch schema for {target_model}: {e}")

        # Step 3: Dynamic few-shot from ChromaDB training data
        dynamic_examples = ""
        if self.vs and self.vs.training_data.count() > 0:
            similar = self.vs.find_similar_training(question, n=3)
            if similar:
                lines = ["LEARNED EXAMPLES FROM YOUR DATA (use these patterns):"]
                for ex in similar:
                    if ex["similarity"] > 0.5 and ex["correct_query"]:
                        lines.append(f"Q: {ex['question']}")
                        lines.append(f"A: {ex['correct_query']}")
                if len(lines) > 1:
                    dynamic_examples = "\n".join(lines) + "\n"

        system = QUERY_PROMPT_TEMPLATE.format(
            today            = today,
            month_start      = month_start,
            year_start       = year_start,
            model            = target_model or "?",
            model_schema     = model_schema or "  (schema unavailable — use your best judgement)",
            dynamic_examples = dynamic_examples,
        )

        generated_query = None
        odoo_model      = target_model or ""
        operation       = "unknown"
        record_count    = 0
        status          = "success"
        error_message   = ""
        answer          = ""

        try:
            # Step 4: Generate JSON query
            raw = self.llm.chat(
                system=system, user_message=question, max_tokens=500
            )
            print(f"\n[ENGINE] Raw LLM: {raw}\n")

            query           = self._parse_json(raw)
            generated_query = query
            operation       = query.get("operation", "unknown")
            odoo_model      = query.get("model", target_model or "")
            data            = []

            # Step 5: Execute
            if operation == "search_read":
                data = self.connector.search_read(
                    model  = odoo_model,
                    domain = query.get("domain", []),
                    fields = query.get("fields", []),
                    limit  = query.get("limit", 20),
                    order  = query.get("order", ""),
                )
                record_count = len(data)

            elif operation == "read_group":
                raw_data = self.connector.read_group(
                    model   = odoo_model,
                    domain  = query.get("domain", []),
                    fields  = query.get("fields", []),
                    groupby = query.get("groupby", []),
                    limit   = query.get("limit", 20),
                    orderby = query.get("order", ""),
                )
                data = [
                    {k: v for k, v in r.items()
                     if not k.startswith("__") and k != "id"}
                    for r in raw_data
                ]
                record_count = len(data)

            elif operation == "search_count":
                record_count = self.connector.search_count(
                    model=odoo_model, domain=query.get("domain", [])
                )
                data = [{"count": record_count}]

            # Step 6: Format
            if not data:
                answer = f"No records found in **{odoo_model}** matching those filters."
                status = "empty"
            else:
                format_q = (
                    f'User asked: "{question}"\n'
                    f"Model: {odoo_model} | Records: {record_count}\n"
                    f"Data:\n{json.dumps(data[:60], indent=2, default=str)}\n\n"
                    f"Format this into a clear answer."
                )
                answer = self.llm.chat(
                    system=FORMAT_PROMPT, user_message=format_q, max_tokens=1500,
                )

        except Exception as e:
            status        = "error"
            error_message = str(e)
            answer        = f"⚠️ **Error:** {error_message}"
            print(f"[ENGINE] Error: {e}")

        execution_ms = int((time.time() - t_start) * 1000)

        # Step 7: Log to ChromaDB
        log_id = None
        if self.vs:
            log_id = self.vs.log_query(
                question        = question,
                answer          = answer,
                generated_query = generated_query,
                odoo_model      = odoo_model,
                operation       = operation,
                record_count    = record_count,
                execution_ms    = execution_ms,
                llm_provider    = llm_provider,
                status          = status,
                error_message   = error_message,
                session_id      = self.session_id or "",
            )
            self._query_count += 1
            if self.session_id:
                self.vs.update_session_count(self.session_id, self._query_count)

        return {
            "answer":       answer,
            "model":        odoo_model,
            "record_count": record_count,
            "log_id":       log_id,
            "query":        generated_query,
            "status":       status,
            "execution_ms": execution_ms,
        }

    def _parse_json(self, raw: str) -> dict:
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        m = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', raw)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s != -1 and e > s:
            try:
                return json.loads(raw[s:e])
            except json.JSONDecodeError:
                pass
        raise ValueError(
            f"Could not parse LLM response as JSON.\n"
            f"Response:\n{raw[:400]}\n\nTry rephrasing your question."
        )