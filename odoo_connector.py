"""
odoo_connector.py — Odoo JSON-RPC interface
Uses /web/dataset/call_kw (NOT xmlrpc — removed in Odoo v20)

Security model:
  READ  (search_read, read_group, search_count, fields_get) → ANY model, no restriction
  WRITE (create, write)                                     → WRITE_BLOCKED_MODELS enforced
  The logged-in Odoo user's own RBAC is the real security layer for reads.
"""

import requests

# ── Write-only safety guard ────────────────────────────────────────────────────
# AI must NEVER write to these models regardless of user permissions.
# Read operations on all models are unrestricted — Odoo RBAC handles that.
WRITE_BLOCKED_MODELS = {
    "res.users",            # never touch user accounts
    "ir.config_parameter",  # system settings
    "ir.rule",              # access rules
    "ir.model.access",      # ACL table
    "base.automation",      # automated actions
    "ir.cron",              # scheduled jobs
    "ir.module.module",     # installed modules
}

# Field types worth showing to the LLM (skip binary, html noise etc.)
USEFUL_FIELD_TYPES = {
    "char", "text", "integer", "float", "monetary",
    "boolean", "date", "datetime", "selection",
    "many2one", "one2many", "many2many",
}


class OdooConnector:

    def __init__(self, url: str, db: str, username: str, password: str):
        self.url      = url.rstrip("/")
        self.db       = db
        self.username = username
        self.password = password
        self.uid      = None
        self.session  = requests.Session()
        self._fields_cache: dict = {}   # model → fields dict
        self._login()

    # ── Auth ───────────────────────────────────────────────────────────────────

    def _login(self):
        resp = self.session.post(
            f"{self.url}/web/session/authenticate",
            json={
                "jsonrpc": "2.0", "method": "call", "id": 1,
                "params": {
                    "db": self.db, "login": self.username, "password": self.password,
                }
            },
            timeout=15,
        )
        result = resp.json().get("result", {})
        self.uid = result.get("uid")
        if not self.uid:
            raise ConnectionError(
                f"Odoo authentication failed.\nResponse: {resp.json()}"
            )

    def _call(self, model: str, method: str, args: list, kwargs: dict = None):
        payload = {
            "jsonrpc": "2.0", "method": "call", "id": 2,
            "params": {
                "model": model, "method": method,
                "args": args, "kwargs": kwargs or {},
            }
        }
        resp = self.session.post(
            f"{self.url}/web/dataset/call_kw",
            json=payload, timeout=30,
        )
        body = resp.json()
        if "error" in body:
            msg = body["error"].get("data", {}).get("message", str(body["error"]))
            raise RuntimeError(f"Odoo JSON-RPC error: {msg}")
        return body.get("result")

    def _check_write(self, model: str):
        """Block writes to sensitive system models only."""
        if model in WRITE_BLOCKED_MODELS:
            raise PermissionError(
                f"Model '{model}' is blocked from AI write operations for safety."
            )

    # ── Connection test ────────────────────────────────────────────────────────

    def test_connection(self) -> bool:
        try:
            result = self._call(
                "res.users", "search_read",
                [[["id", "=", self.uid]]],
                {"fields": ["name"], "limit": 1}
            )
            return bool(result)
        except Exception:
            return False

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL DISCOVERY — no restriction, works for any installed model
    # ══════════════════════════════════════════════════════════════════════════

    def model_exists(self, model: str) -> bool:
        """Check whether a model is installed in this Odoo instance."""
        try:
            result = self._call(
                "ir.model", "search_read",
                [[["model", "=", model]]],
                {"fields": ["model"], "limit": 1}
            )
            return bool(result)
        except Exception:
            return False

    def list_all_models(self, keyword: str = "") -> list[dict]:
        """
        Return all installed models, optionally filtered by keyword.
        Returns list of {"model": "sale.order", "name": "Sales Order"}
        """
        domain = [["model", "ilike", keyword]] if keyword else [[]]
        result = self._call(
            "ir.model", "search_read",
            [domain],
            {"fields": ["model", "name"], "order": "model asc", "limit": 1000}
        )
        return [{"model": r["model"], "name": r["name"]} for r in result]

    # ══════════════════════════════════════════════════════════════════════════
    # FIELD INTROSPECTION — no restriction, any model
    # ══════════════════════════════════════════════════════════════════════════

    def get_fields(self, model: str, filter_useful: bool = True) -> dict:
        """
        Return all fields of any Odoo model via fields_get().
        No model whitelist — works for any installed model.
        Results cached per session.

        filter_useful=True  → stored fields with meaningful types only
        filter_useful=False → every field including computed/binary/private
        """
        cache_key = f"{model}__{filter_useful}"
        if cache_key in self._fields_cache:
            return self._fields_cache[cache_key]

        raw = self._call(
            model, "fields_get", [],
            {
                "attributes": [
                    "string", "type", "required", "store",
                    "readonly", "relation", "selection", "help",
                ]
            }
        )

        if filter_useful:
            raw = {
                name: meta for name, meta in raw.items()
                if meta.get("type") in USEFUL_FIELD_TYPES
                and meta.get("store", True)
                and not name.startswith("_")
            }

        self._fields_cache[cache_key] = raw
        return raw

    def get_fields_summary(self, model: str) -> str:
        """
        Compact text summary of all fields for a model.
        Injected into LLM prompts so it knows the real field names.

        Example:
            name (char) ★ — Order Reference
            partner_id (many2one → res.partner) — Customer
            state (selection: draft|sent|sale|done|cancel) — Status
        """
        fields = self.get_fields(model, filter_useful=True)
        lines  = []

        for fname, meta in sorted(fields.items()):
            ftype    = meta.get("type", "?")
            label    = meta.get("string", fname)
            required = " ★" if meta.get("required") else ""
            relation = meta.get("relation", "")
            choices  = meta.get("selection", [])

            if ftype == "many2one" and relation:
                type_str = f"many2one → {relation}"
            elif ftype in ("one2many", "many2many") and relation:
                type_str = f"{ftype} → {relation}"
            elif ftype == "selection" and choices:
                vals     = "|".join(str(v) for v, _ in choices)
                type_str = f"selection: {vals}"
            else:
                type_str = ftype

            lines.append(f"  {fname} ({type_str}){required} — {label}")

        return "\n".join(lines)

    def get_fields_for_prompt(self, models: list[str]) -> str:
        """
        Build a multi-model schema string for the LLM prompt.
        Only fetches the models you pass in — used for dynamic per-question loading.
        """
        sections = []
        for model in models:
            try:
                summary = self.get_fields_summary(model)
                sections.append(f"MODEL: {model}\n{summary}")
            except Exception as e:
                sections.append(f"MODEL: {model}\n  [unavailable: {e}]")
        return "\n\n".join(sections)

    def get_all_models_summary(self, model_list: list[str] = None) -> str:
        """
        Build schema string for all models in model_list.
        If model_list is None, fetches from Odoo's ir.model (all installed models).
        Cached after first call.
        """
        if model_list is None:
            installed = self.list_all_models()
            model_list = [m["model"] for m in installed]

        return self.get_fields_for_prompt(model_list)

    def get_selection_values(self, model: str, field: str) -> list:
        """Return valid selection values for a field. No model restriction."""
        fields     = self.get_fields(model, filter_useful=False)
        field_meta = fields.get(field, {})
        return field_meta.get("selection", [])

    def clear_fields_cache(self):
        """Force re-fetch after Odoo upgrade or module install."""
        self._fields_cache.clear()

    # ══════════════════════════════════════════════════════════════════════════
    # READ OPERATIONS — no model restriction
    # ══════════════════════════════════════════════════════════════════════════

    def search_read(self, model: str, domain: list, fields: list,
                    limit: int = 20, order: str = "") -> list:
        return self._call(model, "search_read", [domain], {
            "fields": fields, "limit": limit, "order": order,
        })

    def read_group(self, model: str, domain: list, fields: list,
                   groupby: list, limit: int = 20, orderby: str = "") -> list:
        return self._call(model, "read_group", [domain, fields, groupby], {
            "limit": limit, "orderby": orderby,
        })

    def search_count(self, model: str, domain: list) -> int:
        return self._call(model, "search_count", [domain])

    # ══════════════════════════════════════════════════════════════════════════
    # WRITE OPERATIONS — blocked models list enforced
    # ══════════════════════════════════════════════════════════════════════════

    def create(self, model: str, vals: dict) -> int:
        self._check_write(model)
        return self._call(model, "create", [vals])

    def write(self, model: str, ids: list, vals: dict) -> bool:
        self._check_write(model)
        return self._call(model, "write", [ids, vals])