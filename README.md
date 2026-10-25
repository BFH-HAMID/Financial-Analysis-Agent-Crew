# 📈 Financial Analysis Agent Crew

A production-shaped, **multi-agent financial research app**: three specialised AI agents
(Market Researcher, Data Analyst, Report Writer) are coordinated by an orchestrator with a
**revision loop**. Give it a ticker and a date range and it produces a news + sentiment
brief, a full KPI analysis, interactive charts and a **downloadable PDF report** — all in
one Streamlit interface.

Built with **[agno](https://github.com/agno-agi/agno)**, **Gemini**, **Streamlit**,
**yfinance**, **Plotly**, **reportlab** and a finance-tuned **VADER** sentiment model.

---

## Table of contents

1. [Features](#features)
2. [How it works](#how-it-works)
3. [Project structure](#project-structure)
4. [Quick start](#quick-start)
5. [Configuration](#configuration)
6. [Using the app](#using-the-app)
7. [Command line & automation](#command-line--automation)
8. [Docker](#docker)
9. [Deployment](#deployment)
10. [Testing & code quality](#testing--code-quality)
11. [Design decisions & error handling](#design-decisions--error-handling)
12. [Troubleshooting](#troubleshooting)
13. [Disclaimer](#disclaimer)

---

## Features

| Area | What you get |
| --- | --- |
| **UI** | Ticker + date-range form, quick date presets, live progress bar, metric cards, six result tabs, PDF/Markdown/JSON downloads, collapsible **Agent Conversation Log** |
| **Agents** | Market Researcher (news + sentiment), Data Analyst (KPIs + charts), Report Writer (report), coordinated by an agno `Team` in `coordinate` mode |
| **Orchestration** | Sequential delegation **plus a revision loop**: the orchestrator reviews the draft deterministically and sends it back to the writer with specific, actionable instructions |
| **Market data** | yfinance OHLCV history, company profile, benchmark comparison (`^GSPC`) |
| **KPIs** | ROI, annualised return, volatility, Sharpe, max drawdown, RSI(14), SMA 20/50/200, momentum 20/60d, P/E, forward P/E, EPS, beta, dividend yield, market cap, volume trend, 52-week range position |
| **News** | DuckDuckGo news → DuckDuckGo web → NewsAPI (optional key) with de-duplication and a date window |
| **Sentiment** | VADER + a finance lexicon (`guidance cut`, `downgrade`, `record high`, …) with per-headline scores and an aggregate label |
| **Charts** | Plotly price/SMA, volume, RSI and KPI bar charts; rendered to PNG for the PDF (kaleido when available, matplotlib otherwise) |
| **PDF** | Cover with recommendation badge and confidence bar, executive summary, company snapshot, news table, sentiment box, 23-row KPI table, charts, risk assessment, recommendation, confidence, methodology, disclaimer, optional conversation-log appendix |
| **Robustness** | Typed errors, graceful degradation at every step, retries, rotating logs and an **offline deterministic crew** when no API key is present |
| **Ops** | Dockerfile + compose, Makefile, GitHub Actions (CI + scheduled reports), Telegram/email delivery hooks |

---

## How it works

```
                    ┌──────────────────────────────┐
   Ticker + range → │   Orchestrator (agno Team)   │
                    └───────────────┬──────────────┘
                                    │ 1. delegate
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
   ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
   │  Market Researcher │ │    Data Analyst    │ │   Report Writer    │
   │  news + sentiment  │ │  KPIs + charts     │ │  Markdown report   │
   └─────────┬──────────┘ └─────────┬──────────┘ └─────────┬──────────┘
             │ search_stock_news    │ get_stock_snapshot    │
             │ analyze_news_sentiment│ get_financial_kpis   │
             │                      │ get_market_chart      │
             └──────────────────────┴───────────────────────┘
                                    │ 2. draft
                                    ▼
                     ┌───────────────────────────────┐
                     │ Orchestrator review (QA)      │
                     │ missing sections / no score / │
                     │ contradictions / demo data    │
                     └──────────────┬────────────────┘
                          fails ────►│◄──── 3. revision loop
                                    │      (max N rounds)
                          passes ───▼
                     ┌───────────────────────────────┐
                     │  Report + charts + PDF + log  │
                     └───────────────────────────────┘
```

**Two execution paths, one contract.** `AnalysisResult` is identical whether the crew ran
with Gemini (live) or without it (offline). Market data, news, sentiment, charts and the PDF
are always produced by deterministic code — the LLM adds narrative and judgement. This means:

* a missing/expired API key never breaks the app;
* the PDF always contains real (or clearly labelled demo) tables and images;
* the whole project is testable offline in CI.

### The revision loop

After the writer's draft, `agents.orchestrator.review_draft()` checks:

1. all required sections exist (Executive Summary, KPI table, Recommendation, Confidence, Disclaimer);
2. a valid recommendation label is present;
3. a numeric confidence score is present;
4. at least eight concrete figures are cited;
5. sentiment and recommendation do not contradict each other without explanation;
6. synthetic data is disclosed and confidence is not inflated above 85.

If any check fails, the failures become a precise instruction set sent back to the Report
Writer (up to `MAX_REVISION_ROUNDS` times). The loop is deterministic, so it can't run away
with cost. Anything still missing is filled from the deterministic report, in an explicitly
labelled appendix.

---

## Project structure

```
Financial-Analysis-Agent-Crew/
├── app.py                     # Streamlit entry point (streamlit run app.py)
├── agents/
│   ├── __init__.py
│   ├── base.py                # response-text extraction, tool introspection
│   ├── llm.py                 # Gemini model factory (single source of truth)
│   ├── researcher.py          # Market Researcher agent
│   ├── analyst.py             # Data Analyst agent
│   ├── writer.py              # Report Writer agent
│   ├── offline_crew.py        # deterministic crew (no API key needed) + heuristic verdict
│   └── orchestrator.py        # Team orchestration, revision loop, assembly
├── config/
│   └── settings.py            # env-driven settings, paths, constants
├── prompts/
│   └── prompts.py             # every agent instruction + report contract
├── tools/
│   ├── finance_tools.py       # yfinance data, KPI maths, agno tool wrappers
│   ├── news_tools.py          # DuckDuckGo / NewsAPI search, cleaning, fallbacks
│   ├── sentiment.py           # VADER + finance lexicon, aggregation
│   ├── chart_tools.py         # Plotly figures → PNG (kaleido or matplotlib)
│   └── demo_data.py           # labelled synthetic data for offline demos
├── reports/
│   ├── pdf_generator.py       # reportlab PDF (cover, tables, charts, appendix)
│   └── output/                # generated reports, charts and PDFs (git-ignored)
├── utils/
│   ├── logger.py              # rotating logs + ConversationLog
│   ├── parsing.py             # ticker/date validation, LLM output parsing
│   └── formatting.py          # currency / percent / compact formatting
├── scripts/
│   └── daily_report.py        # headless batch runs, Telegram / email delivery
├── tests/                     # 150+ offline tests (pytest, incl. Streamlit AppTest)
├── .github/workflows/         # ci.yml + daily-report.yml
├── Dockerfile, docker-compose.yml, Makefile
├── requirements.txt, requirements-dev.txt
├── .env.example, pytest.ini, .streamlit/config.toml
└── README.md
```

---

## Quick start

### 1. Requirements

* Python **3.10+** (tested on 3.11)
* A Google **Gemini API key** — optional but recommended
  ([get one free](https://aistudio.google.com/app/apikey))

### 2. Install

```bash
git clone https://github.com/BFH-HAMID/Financial-Analysis-Agent-Crew.git
cd Financial-Analysis-Agent-Crew

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# then edit .env and set GOOGLE_API_KEY=...
```

No key? Skip this step. The app starts in **offline crew** mode: full report, charts and PDF,
no LLM narrative.

### 4. Run

```bash
streamlit run app.py
```

Open <http://localhost:8501>, enter a ticker (e.g. `AAPL`), pick a date range and press
**Analyze**. Outputs land in `reports/output/`.

```bash
make install   # venv + dependencies
make run       # start the app
make test      # run the test-suite (offline)
make demo      # CLI demo report, no API key required
```

---

## Configuration

Everything is environment-driven (see `config/settings.py`). `.env` values are loaded once at
import; real environment variables always win, so Docker/Streamlit Cloud secrets are safe.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOOGLE_API_KEY` | – | Gemini key. Absent ⇒ offline crew |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Any Gemini model id available to your key |
| `GEMINI_TEMPERATURE` | `0.3` | Sampling temperature |
| `LLM_TIMEOUT_SECONDS` | `120` | Per-request timeout |
| `NEWS_API_KEY` | – | Optional NewsAPI fallback (DuckDuckGo needs no key) |
| `MAX_REVISION_ROUNDS` | `2` | Revision-loop budget |
| `TEAM_MODE` | `coordinate` | agno team mode (`coordinate`, `route`, `broadcast`, `tasks`) |
| `TEAM_MAX_ITERATIONS` | `6` | Safety cap on team iterations |
| `SENTIMENT_BACKEND` | `lexicon` | `lexicon` (VADER) or `llm` |
| `RISK_FREE_RATE` | `0.04` | Sharpe/risk-free assumption |
| `TRADING_DAYS` | `252` | Annualisation factor |
| `MAX_NEWS_RESULTS` | `8` | Headlines per run |
| `NEWS_LOOKBACK_DAYS` | `30` | News freshness window |
| `ALLOW_DEMO_DATA` | `true` | Use labelled synthetic data if providers fail |
| `FORCE_DEMO_DATA` | `false` | Skip live providers entirely (demos, CI) |
| `REPORTS_DIR` / `LOG_DIR` | `reports/output` / `logs` | Output locations |
| `LOG_LEVEL` | `INFO` | Console + rotating-file log level |

> **Demo data is always labelled.** Synthetic runs are marked in the UI banner, in the
> conversation log, on the PDF cover and inside the report body, and they cap the confidence
> score at 65.

---

## Using the app

1. **Sidebar** — connection status, date presets, max revision rounds, force-demo toggle,
   runtime details and recent runs.
2. **Form** — ticker, start date, end date, **Analyze**.
3. **Progress** — staged updates: *Input → Data → Charts → Agents → Review → Files → PDF → Done*.
4. **Results**
   * *Report* — the full Markdown report as produced by the crew;
   * *Charts* — four interactive Plotly figures;
   * *KPI Table* — 23 metrics with interpretation notes;
   * *News & Sentiment* — aggregate scores, observations, headlines and per-headline scores;
   * *Agent Conversation Log* — timestamped agent hand-offs, tool calls and revisions;
   * *Raw data* — the KPI dict, timings and saved file paths.
5. **Downloads** — PDF report, Markdown, KPI JSON, conversation JSON.

---

## Command line & automation

```bash
# Single run, offline (synthetic, labelled)
FORCE_DEMO_DATA=1 python scripts/daily_report.py --tickers AAPL --days 180 --no-llm

# Watchlist with the live crew, PDFs in reports/output/pdfs
python scripts/daily_report.py --tickers AAPL,MSFT,NVDA --days 90

# Explicit window + Telegram summary
python scripts/daily_report.py --tickers TSLA --start 2024-01-01 --end 2024-06-30 --telegram
```

The script writes a machine-readable `reports/output/daily_summary.json` and exits non-zero
if no ticker produced a report (friendly for cron/CI). Telegram needs
`TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`; email needs the `SMTP_*` / `EMAIL_*` variables.

**Cron example** (daily at 07:00):

```cron
0 7 * * * cd /srv/financial-crew && .venv/bin/python scripts/daily_report.py --tickers AAPL,MSFT --days 90 --telegram >> logs/cron.log 2>&1
```

`schedules`/GitHub Actions equivalents are included in `.github/workflows/daily-report.yml`.

---

## Docker

```bash
cp .env.example .env      # add your keys
docker build -t financial-crew .
docker run --rm -p 8501:8501 --env-file .env \
  -v "$PWD/reports/output:/app/reports/output" \
  financial-crew
```

or

```bash
docker compose up --build
```

The image runs as a non-root user, exposes a healthcheck (`/_stcore/health`) and binds
`0.0.0.0:8501`. Charts are rasterised with matplotlib, so the image does **not** need Chrome;
uncomment the `kaleido` lines in the `Dockerfile` only if you want Plotly's native export.

---

## Deployment

* **Streamlit Community Cloud** — point it at `app.py`, then add `GOOGLE_API_KEY` (and
  optionally `NEWS_API_KEY`) under *Settings → Secrets*. The app copies recognised secrets
  into the environment on start-up (`_load_streamlit_secrets`).
* **Render / Railway / Fly.io** — build from the Dockerfile; set the same env vars.
* **Kubernetes** — the container is stateless except for `reports/output` and `logs`; mount a
  volume if you want to keep artefacts.
* **Scheduled reporting** — `.github/workflows/daily-report.yml` runs the CLI daily and
  uploads the PDFs as build artefacts. Add `GOOGLE_API_KEY` (and Telegram secrets) as
  repository secrets to enable it.

---

## Testing & code quality

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q            # 150+ tests, fully offline
python -m pytest --cov=. --cov-report=term-missing
ruff check .                          # lint
ruff format .                         # format
```

The suite covers:

| File | Focus |
| --- | --- |
| `tests/test_finance_tools.py` | RSI/drawdown/volatility maths, demo-data reproducibility, tool JSON contracts, failure handling |
| `tests/test_sentiment.py` | lexicon behaviour, aggregation, edge cases |
| `tests/test_news_tools.py` | provider fallbacks, de-duplication, sorting, date filtering |
| `tests/test_charts_and_pdf.py` | figure construction, PNG export, PDF content/structure, Markdown→PDF rendering |
| `tests/test_orchestrator.py` | end-to-end offline run, required sections, revision QA, artefact writing |
| `tests/test_llm_and_fallbacks.py` | model factory, live-crew failure fallback, draft gap-filling |
| `tests/test_app.py` | real Streamlit app via `AppTest` (startup, validation, full click-through) |
| `tests/test_utils.py` | logging, conversation log, formatting, settings |

Tests never call Gemini or Yahoo: external calls are replaced by fixtures and synthetic data.

---

## Design decisions & error handling

* **`AnalysisResult` as the single contract** — the UI, PDF generator and tests consume the
  same object, so live and offline runs are interchangeable.
* **Deterministic core, LLM on top** — KPIs, charts and PDF never depend on the model
  behaving; that is what makes the app reliable and cheap to test.
* **Everything degrades** — market data → demo data; news → demo headlines; LLM → offline
  crew; kaleido → matplotlib; PDF → analysis still returned (with a warning). Each fallback is
  logged and surfaced in the UI and the PDF.
* **No fabricated numbers** — agents are instructed (and the orchestrator verifies) to cite
  tool output only, and synthetic data is labelled at every level.
* **Typed errors** — `InputValidationError`, `FinanceDataError`, `NewsError`, `ChartError`,
  `LLMConfigurationError`, `PDF generation RuntimeError`; the UI shows the message, the log
  keeps the traceback.
* **Logging** — rotating file (`logs/financial_crew.log`) + console, namespaced per module,
  plus the structured `ConversationLog` that feeds the UI expander and the PDF appendix.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `GOOGLE_API_KEY is not set` warning | Add the key to `.env`, or accept the offline crew |
| `No market data` / empty charts | Check the ticker (Yahoo symbols, e.g. `BRK-B`, `0700.HK`), the date range, or your network; demo mode covers outages |
| Report says **SYNTHETIC DEMO DATA** | Live providers were unreachable (or `FORCE_DEMO_DATA=1`). Set `ALLOW_DEMO_DATA=false` to fail loudly instead |
| No charts inside the PDF | Check `reports/output/charts/*.png`; install `kaleido` + Chrome for Plotly export, or keep matplotlib |
| `Kaleido requires Google Chrome` in logs | Harmless — the matplotlib renderer takes over |
| PDF generation failed | Look at the traceback in `logs/financial_crew.log`; the Markdown report is still saved |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Slow first run | `yfinance` fetches history + profile; subsequent runs are faster |

---

## Disclaimer

This project generates **automated research for educational purposes**. It is **not**
investment advice, and nothing it produces is a recommendation to buy or sell a security. Data
may be delayed, incomplete, or synthetic (clearly labelled). Past performance does not
guarantee future results. Always verify figures against primary sources and consult a
licensed financial professional.

Licensed under the terms in [LICENSE](LICENSE).
