"""
ai_engine.py
────────────
Step 1: Question → LLM → JSON query
Step 2: Execute JSON query against Odoo via JSON-RPC
Step 3: Raw data → LLM → formatted answer
Step 4: Log everything to ChromaDB vector store
Step 5 (bonus): Inject similar past correct queries as dynamic few-shot examples
"""

import json
import re
import time
from datetime import datetime

from odoo_connector import OdooConnector
from llm_client import GroqClient


QUERY_PROMPT = """You generate Odoo ERP API queries from natural language questions.

TODAY: {today}
MONTH START: {month_start}
YEAR START: {year_start}

{dynamic_examples}

MODELS:
- sale.order: name, partner_id, amount_total, date_order, state(draft/sent/sale/done/cancel), user_id
- account.move: name, partner_id, amount_total, amount_residual, payment_state(not_paid/paid/partial), invoice_date, invoice_date_due, move_type(out_invoice=customer invoice, in_invoice=vendor bill), state(draft/posted/cancel)
- stock.quant: product_id, quantity, reserved_quantity, location_id
- res.partner: name, email, phone, city, customer_rank, supplier_rank
- product.template: name, default_code, list_price, standard_price, categ_id, active
- purchase.order: name, partner_id, amount_total, date_order, state(draft/sent/purchase/done/cancel)
- hr.employee: name, department_id, job_title, active
- crm.lead: name, partner_id, stage_id, probability, expected_revenue, user_id
- sale.order.line: order_id, product_id, product_uom_qty, price_unit, price_subtotal

OPERATIONS:
1. search_read — list records
2. read_group  — totals / aggregations / group by
3. search_count — count only

DOMAIN OPERATORS: = != > < >= <= like ilike in not in
DATE FORMAT: "YYYY-MM-DD"

OUTPUT: return ONLY a raw JSON object — no explanation, no markdown, no backticks.

For search_read:
{{"operation":"search_read","model":"MODEL","domain":[...],"fields":["f1","f2"],"order":"field desc","limit":20}}

For read_group:
{{"operation":"read_group","model":"MODEL","domain":[...],"fields":["field","amount:sum"],"groupby":["field"],"order":"amount desc","limit":10}}

For search_count:
{{"operation":"search_count","model":"MODEL","domain":[...]}}

BUILT-IN EXAMPLES:
Q: top 5 customers by sales this month
A: {{"operation":"read_group","model":"sale.order","domain":[["state","=","sale"],["date_order",">=","{month_start}"]],"fields":["partner_id","amount_total:sum"],"groupby":["partner_id"],"order":"amount_total desc","limit":5}}

Q: unpaid invoices
A: {{"operation":"search_read","model":"account.move","domain":[["move_type","=","out_invoice"],["payment_state","=","not_paid"],["state","=","posted"]],"fields":["name","partner_id","amount_total","amount_residual","invoice_date_due"],"order":"invoice_date_due asc","limit":50}}

Q: how many open sales orders
A: {{"operation":"search_count","model":"sale.order","domain":[["state","=","sale"]]}}

Q: products with low stock
A: {{"operation":"search_read","model":"stock.quant","domain":[["quantity","<",10],["quantity",">",0],["location_id.usage","=","internal"]],"fields":["product_id","quantity","reserved_quantity"],"order":"quantity asc","limit":50}}

Output must start with {{ and end with }}, nothing else."""


FORMAT_PROMPT = """You are a business data analyst. Present Odoo ERP data clearly.

Rules:
- Use markdown tables for lists of records
- Format currency with 2 decimal places and commas
- For fields like [42, "Customer Name"], show only the name
- Give 1-2 key insights at the end
- Be concise and clear"""


class OdooAIEngine:

    def __init__(self, connector: OdooConnector, groq_api_key: str,
                 vector_store=None):
        """
        connector     — OdooConnector instance
        groq_api_key  — Groq API key
        vector_store  — VectorStore instance (optional, None = no logging)
        """
        self.connector    = connector
        self.llm          = GroqClient(groq_api_key)
        self.vs           = vector_store
        self.session_id   = None
        self._query_count = 0

    def set_session(self, session_id: str):
        self.session_id = session_id

    def clear_history(self):
        self._query_count = 0

    def ask(self, question: str, llm_provider: str = "groq") -> dict:
        t_start     = time.time()
        today       = datetime.now().strftime("%Y-%m-%d")
        month_start = datetime.now().strftime("%Y-%m-01")
        year_start  = datetime.now().strftime("%Y-01-01")

        # ── Dynamic few-shot examples from training data ───────────────────────
        # If we have corrected examples in ChromaDB that are similar to this
        # question, inject them into the prompt automatically
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

        system = QUERY_PROMPT.format(
            today=today,
            month_start=month_start,
            year_start=year_start,
            dynamic_examples=dynamic_examples,
        )

        generated_query = None
        odoo_model      = ""
        operation       = "unknown"
        record_count    = 0
        status          = "success"
        error_message   = ""
        answer          = ""

        try:
            # ── Step 1: Generate query ─────────────────────────────────────────
            raw = self.llm.chat(
                system=system, user_message=question, max_tokens=500
            )
            print(f"\n[ENGINE] Raw LLM response: {raw}\n")

            query           = self._parse_json(raw)
            generated_query = query
            operation       = query.get("operation", "unknown")
            odoo_model      = query.get("model", "")
            data            = []

            # ── Step 2: Execute against Odoo ──────────────────────────────────
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

            # ── Step 3: Format answer ──────────────────────────────────────────
            if not data:
                answer = (
                    f"No records found in **{odoo_model}** "
                    f"matching those filters."
                )
                status = "empty"
            else:
                format_q = (
                    f'User asked: "{question}"\n'
                    f"Model: {odoo_model} | Records: {record_count}\n"
                    f"Data:\n{json.dumps(data[:60], indent=2, default=str)}\n\n"
                    f"Format this into a clear answer."
                )
                answer = self.llm.chat(
                    system=FORMAT_PROMPT,
                    user_message=format_q,
                    max_tokens=1500,
                )

        except Exception as e:
            status        = "error"
            error_message = str(e)
            answer        = f"⚠️ **Error:** {error_message}"
            print(f"[ENGINE] Error: {e}")

        execution_ms = int((time.time() - t_start) * 1000)

        # ── Step 4: Log to ChromaDB ────────────────────────────────────────────
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
        if '"operation"' in raw and '"model"' in raw:
            try:
                return json.loads("{" + raw.strip().strip(",") + "}")
            except json.JSONDecodeError:
                pass
        raise ValueError(
            f"Could not parse LLM response as JSON.\n"
            f"Response:\n{raw[:400]}\n\nTry rephrasing your question."
        )
