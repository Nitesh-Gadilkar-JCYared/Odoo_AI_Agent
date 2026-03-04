"""
odoo_connector.py — Odoo JSON-RPC interface
Uses /web/dataset/call_kw (NOT xmlrpc — removed in Odoo v20)
"""

import json
import requests

ALLOWED_MODELS = {
    # Business models
    "sale.order", "sale.order.line",
    "account.move", "account.move.line", "account.payment",
    "stock.quant", "stock.picking", "stock.move",
    "res.partner", "product.template", "product.product",
    "purchase.order", "purchase.order.line",
    "hr.employee", "hr.department",
    "project.project", "project.task",
    "crm.lead", "mrp.production",
    # AI logging models (always allowed)
    "ai.query.log",
    "ai.chat.session",
}


class OdooConnector:

    def __init__(self, url: str, db: str, username: str, password: str):
        self.url      = url.rstrip("/")
        self.db       = db
        self.username = username
        self.password = password
        self.uid      = None
        self.session  = requests.Session()
        self._login()

    def _login(self):
        resp = self.session.post(
            f"{self.url}/web/session/authenticate",
            json={
                "jsonrpc": "2.0",
                "method":  "call",
                "id":      1,
                "params": {
                    "db":       self.db,
                    "login":    self.username,
                    "password": self.password,
                }
            },
            timeout=15,
        )
        result = resp.json().get("result", {})
        self.uid = result.get("uid")
        if not self.uid:
            raise ConnectionError(
                "Odoo authentication failed. Check your credentials.\n"
                f"Response: {resp.json()}"
            )

    def _call(self, model: str, method: str, args: list, kwargs: dict = None) -> any:
        """Core JSON-RPC call to /web/dataset/call_kw"""
        payload = {
            "jsonrpc": "2.0",
            "method":  "call",
            "id":      2,
            "params": {
                "model":  model,
                "method": method,
                "args":   args,
                "kwargs": kwargs or {},
            }
        }
        resp = self.session.post(
            f"{self.url}/web/dataset/call_kw",
            json=payload,
            timeout=30,
        )
        body = resp.json()
        if "error" in body:
            msg = body["error"].get("data", {}).get("message", str(body["error"]))
            raise RuntimeError(f"Odoo JSON-RPC error: {msg}")
        return body.get("result")

    def _check_model(self, model: str):
        if model not in ALLOWED_MODELS:
            raise PermissionError(
                f"Model '{model}' is not in the allowed list. "
                f"Add it to ALLOWED_MODELS in odoo_connector.py"
            )

    def test_connection(self) -> bool:
        try:
            result = self._call("res.users", "search_read",
                                [[["id", "=", self.uid]]],
                                {"fields": ["name"], "limit": 1})
            return bool(result)
        except Exception:
            return False

    # ── Read operations ────────────────────────────────────────────────────────

    def search_read(self, model: str, domain: list, fields: list,
                    limit: int = 20, order: str = "") -> list:
        self._check_model(model)
        return self._call(model, "search_read", [domain], {
            "fields": fields,
            "limit":  limit,
            "order":  order,
        })

    def read_group(self, model: str, domain: list, fields: list,
                   groupby: list, limit: int = 20, orderby: str = "") -> list:
        self._check_model(model)
        return self._call(model, "read_group", [domain, fields, groupby], {
            "limit":   limit,
            "orderby": orderby,
        })

    def search_count(self, model: str, domain: list) -> int:
        self._check_model(model)
        return self._call(model, "search_count", [domain])

    # ── Write operations (used by logger only) ─────────────────────────────────

    def create(self, model: str, vals: dict) -> int:
        """Create a record — used by OdooLogger for ai.query.log / ai.chat.session"""
        self._check_model(model)
        return self._call(model, "create", [vals])

    def write(self, model: str, ids: list, vals: dict) -> bool:
        """Write to existing records — used by OdooLogger for feedback updates"""
        self._check_model(model)
        return self._call(model, "write", [ids, vals])
