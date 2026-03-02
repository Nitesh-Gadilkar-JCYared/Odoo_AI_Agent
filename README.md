# Odoo AI Assistant — Streamlit App

Natural language chatbot for your Odoo ERP data, powered by Claude AI.

---

## 📁 Project Structure

```
odoo_ai_bot/
├── app.py              ← Main Streamlit UI
├── ai_engine.py        ← AI brain (query generation + formatting)
├── odoo_connector.py   ← Odoo XML-RPC interface
├── config.py           ← Configuration
├── requirements.txt    ← Python dependencies
└── .env.example        ← Environment variables template
```

---

## ⚙️ Setup (5 minutes)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials (choose one method)

**Method A — .env file (recommended)**
```bash
cp .env.example .env
# Edit .env with your values
```

**Method B — Enter in UI sidebar**
Just run the app and fill in credentials in the left sidebar.

### 3. Run the app

```bash
streamlit run app.py
```

Open browser at: http://localhost:8501

---

## 🔌 Odoo Requirements

- Odoo 15, 16, or 17 (XML-RPC is enabled by default)
- A user account with read access to the models you want to query
- No extra Odoo modules required

---

## 💬 Example Questions You Can Ask

**Sales:**
- "Top 10 sales orders this month by amount"
- "Which salesperson has the highest revenue this year?"
- "Show me all cancelled orders this week"
- "Sales summary by product category for Q1"

**Finance:**
- "Show all unpaid invoices older than 30 days"
- "Total revenue this month vs last month"
- "Which customers owe us more than $10,000?"
- "All vendor bills due this week"

**Inventory:**
- "Products with stock below 20 units"
- "What is our total inventory value?"
- "Which products have zero stock?"
- "Pending delivery orders for this week"

**Customers:**
- "Top 5 customers by purchase volume"
- "Customers who haven't ordered in 90 days"
- "New customers added this month"

**HR:**
- "How many employees in each department?"
- "List all employees in the Sales department"

---

## 🔒 Security Notes

- The app is read-only by default — it cannot create, update, or delete Odoo records
- All queries go through a whitelisted model list in `odoo_connector.py`
- Credentials are never stored — entered per session
- Uses the Odoo user's existing permissions (if user can't see it in Odoo, AI can't see it either)

---

## 🚀 Deployment Options

**Local (development):**
```bash
streamlit run app.py
```

**Server (production):**
```bash
# With systemd service or screen
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

**Docker:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

---

## 🛠️ Extending the Bot

**Add a new Odoo model:**
In `odoo_connector.py`, add the model to `ALLOWED_READ_MODELS`.
In `ai_engine.py`, add the model and its fields to `QUERY_SYSTEM_PROMPT`.

**Add a quick query button:**
In `app.py`, add an entry to the `quick_queries` dict in the sidebar.

**Improve query accuracy:**
Add more few-shot examples to `QUERY_SYSTEM_PROMPT` in `ai_engine.py`.
