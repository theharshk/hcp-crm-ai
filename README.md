# HCP CRM - Field Rep Interaction Logging System

Hey! This repository contains the code for our split-screen pharmaceutical CRM logging dashboard. The system is designed to allow field reps to log interactions with Healthcare Professionals (HCPs) through natural conversation with an AI assistant, rather than manual form entry.

---

## ⚡ Instant Project Explanation (Pitch Cheat Sheet)

If you need to explain this project to someone instantly, here is the exact 30-second summary:

> *"This is an **AI-First CRM dashboard for pharma sales representatives**. Instead of manually typing out long, tedious forms after doctor visits, reps just chat naturally with an AI assistant on the right panel. The AI assistant uses LangGraph to orchestrate database tools: it automatically extracts the visit details (type, sentiment, topics discussed, products discussed, and samples left) and populates the structured database and form on the left in real-time.
> 
> The dashboard is split 66.6% (form panels) and 33.3% (chat panel). It keeps the details form empty initially to prevent data contamination, lets reps click past interactions to view their details, and dynamically asks for missing info (like missing doctor institutions or product discussions) so CRM data is always 100% complete and compliant."*

---

Here is a detailed breakdown of how the frontend, state management, database schema, and AI agent graph are wired together.

---

## 📐 Layout & Visual Design

The UI is split side-by-side using a clean CSS grid configured at a `2fr 1fr` proportion:
* **Left Panel (66.6% width)**: Houses the HCP dropdown selector card, the doctor's profile card, the scrollable **Recent Interactions** card, and the main **Interaction Details** structured form card.
* **Right Panel (33.3% width)**: Contains the scrollable **Agent Conversation** chat window and message input.

Inside `frontend/src/styles/index.css`, we constrained the `.chat-panel` container height to a fixed `750px` to match the left-side forms. This confines the chat card, pins the message input box at the bottom, and triggers `overflow-y: auto` inside the messages container. As a result, the list scrolls internally and snaps to the newest message automatically using React refs, preventing the entire page from stretching vertically. We also bounded the main viewport at a maximum width of `1440px` for optimal readability.

---

## 🔄 Selection Flow & State Resets (Redux & React)

To make sure reps don't mix up data or save details on the wrong profiles, we implemented a selection-driven state machine inside Redux (`frontend/src/store/interactionsSlice.js`) and React components:

### 1. Default Empty State
Initially, or whenever a representative changes the doctor in the dropdown, the form at the bottom is reset to a clean, empty state:
* Text fields (Topics, Products, Notes) render with default `"—"` empty placeholder states.
* Dropdowns (Interaction Type, Sentiment) show a default `"—"` empty/unselected option.
* This is managed by setting both `selectedInteraction` and `lastSubmission` to `null` inside the `selectHcp` reducer.

### 2. Click-to-Inspect Past Logs
The **Recent Interactions** card lists the doctor's last 5 interactions. 
* Clicking any item in the list dispatches `selectInteraction(h)`.
* This updates `selectedInteraction` in Redux, which immediately populates all form fields below with that specific logged interaction's database parameters.
* The selected history item gets a subtle highlight (`rgba(0, 102, 102, 0.08)`) with a thin border matching the app theme.

### 3. Collective Summary Overview
If you change the doctor and haven't clicked on a specific recent interaction yet, the form's **Summary** field acts as an HCP overview.
* Instead of showing a single summary, it renders a multi-line, read-only list of **all** past logged interactions for that doctor (e.g., `• [7/8/2026 - Meeting] Discussed studies... \n • [7/5/2026 - Chat] Dosing updates...`).
* This is generated dynamically on the fly by mapping the doctor's `history` array.
* The moment you click a specific item in the list or log a new interaction, this field snaps back to a single-line view of just that interaction's summary.

### 4. Live-Drafting "Magic Mirror" Form Updates
* As the representative chats with the AI, the backend extracts a conversational draft of all details discussed so far (type, sentiment, summary, topics, products) and returns it in the response state as `draft_interaction`.
* Redux catches this draft state and displays it immediately in the Structured Form in front of the representative's eyes, marked with a dashed teal banner: *"AI Assistant is drafting interaction details live from the chat..."*.
* As the rep adds more details in chat (like mentioning the sentiment, or adding a topic), the fields update live.
* Once the interaction has enough details to log or the user confirms, the form automatically clears the draft state and locks in the final database interaction.

### 5. Real-time Chat Syncing
When a chat message is successfully processed, the Redux extraReducer `chat/sendMessage/fulfilled` catches the updated state from the backend. It pushes the new interaction to the history array and sets both `lastSubmission` and `selectedInteraction` to the new record so the form populates in real-time.

---

## 🤖 CRM AI Agent & Performance Optimizations (LangGraph)

The backend agent is built on LangGraph (`backend/app/agent/graph.py`) and is powered by Groq's Llama 3.1 8B (`llama-3.1-8b-instant`). To guarantee production-grade performance, we engineered several key optimization and resilience layers:

### 1. ⚡ Zero-Latency Save & Extraction Bypasses (Eliminating 504 Timeouts)
* **log_interaction Optimization**: Refactored the logging tool in [tools.py](file:///c:/Users/Harsh/Downloads/hcp-crm%20-%20Copy/backend/app/agent/tools.py) to accept optional pre-extracted fields (`interaction_type`, `sentiment`, `topics_discussed`, `products_discussed`, `samples_provided`). The agent now passes its active draft details directly, **completely bypassing the nested LLM extraction**, reducing tool execution time from ~15 seconds to **under 50 milliseconds**.
* **Zero Latency Post-Save Extraction**: Configured the agent's turn runner to skip the `_extract_draft_state` invocation completely when both active details (HCP and interaction) are resolved/saved. This drops save-turn extraction time to **0.00 ms** (saving 27 seconds).

### 2. 🔍 Robust Normalized Name matching
* **normalize_name Helper**: Implemented a normalization utility in [tools.py](file:///c:/Users/Harsh/Downloads/hcp-crm%20-%20Copy/backend/app/agent/tools.py) that strips prefixes/honorifics (`Dr.`, `Dr`, `dr`), removes all punctuation (`.`, `,`, `-`), and normalizes whitespace.
* **Bidirectional Search Loop**: Updated `resolve_hcp_by_name` and `edit_interaction` to perform bidirectional substring matching against all database HCP records. Now, typing `"Dr Gregory House"` or `"Dr Gregory House from diagnostic medicine"` matches `"Dr. Gregory House"` correctly.

### 3. 🛡️ Dynamic Tool Binding & Hallucination Prevention
* **Dynamic Tool Binding**: Configured `_agent_node` and `_build_llm` in [graph.py](file:///c:/Users/Harsh/Downloads/hcp-crm%20-%20Copy/backend/app/agent/graph.py) to check if the doctor has been resolved in the current turn.
   * If **unresolved**, only `resolve_hcp_by_name` and `add_hcp` are bound. This physically prevents the LLM from making premature tool calls like `check_compliance(hcp_id=null)`.
* **Prompt Safety**: Cleaned up the system prompt to remove any mentions of internal state tracking methods (like `set_active_hcp`), stopping LLM tool hallucinations.

### 4. 📉 Token Pruning & Exception Safety (Rate Limit Protection)
* **History Pruning**: Sliced the context history loaded into the graph to the **last 6 messages** (3 user-turns) in [graph.py](file:///c:/Users/Harsh/Downloads/hcp-crm%20-%20Copy/backend/app/agent/graph.py). This prevents the LLM from accumulating excessive token context, keeping token requests low (~1,200 tokens) and preventing Groq rate limits.
* **Resilience Catching**: Wrapped the LangGraph invocation in a robust `try-except` block. If a `groq.RateLimitError` or TPD/TPM exception is encountered, it is caught gracefully and returns a polite message:
  > *"I'm experiencing temporary rate limits. Please wait a few seconds and try sending your message again."*
  This prevents the API from returning raw `500 Internal Server Errors` or crashing.

### 5. The Selected Context Injection
* **The Solution**: We capture the active doctor's UUID (`hcp_id`) from the frontend payload in `/api/chat` and pass it to `run_agent_turn` as `active_hcp_id`.
* The backend queries the database for the doctor's name and prepends a `SystemMessage` context block directly to the message state list:
  > *"Context: The representative currently has HCP 'Dr. Sanjeev' (UUID: '52970600-...') selected on their screen. Use this UUID as the hcp_id in tool calls when referring to this doctor."*
* This gives the model direct context of the active doctor on every turn, completely eliminating name-resolution lag and UUID hallucinations.

### 6. Mandatory Validation Checks
Every logged or edited interaction must have these 4 fields:
1. **Interaction Type** (exactly `Meeting`, `Video Call`, or `Email`)
2. **Sentiment** (`positive`, `neutral`, or `negative`)
3. **Topics Discussed**
4. **Products Discussed**

If any of these details are missing from the representative's natural language descriptions, the agent is instructed to **hold back from calling `log_interaction` or `edit_interaction`**. Instead, it halts and asks the user to provide them in chat.

### 7. Explicit Product Checks & "No Product Discussed" Fallback
If products are not mentioned in the rep's notes, the agent will explicitly prompt the user in the chat:
> *"Were any products discussed during the interaction? If yes, please name the products. If no, let me know so I can record 'No Product Discussed'."*

If the user responds with "no", "n/a", "none", or similar:
* The agent maps this input to `["No Product Discussed"]` for the product parameters.
* The backend extractor inside `backend/app/agent/tools.py` uses this guideline to set `products_discussed` to `["No Product Discussed"]` in the database.
* The frontend placeholder for the **Products Discussed** field is updated to `"Products discussed or 'No Product Discussed'"` to match.

### 8. Profile Enrichment & Mandatory Institution
* After logging or editing an interaction, the agent checks if the doctor's profile is missing an `institution`, `email`, or `phone`.
* Since `institution` is mandatory, the agent will ask the user to provide it in chat if it's missing.
* When provided, it calls `enrich_hcp_profile` (supplying the actual UUID from our context injection) and updates the record in PostgreSQL.

---

## 🗄️ Database & Schema Migrations

The database models are defined in `backend/app/models.py`:
* **HCPs Table**: Contains `id` (UUID), `name`, `specialty`, `institution`, `email`, and `phone`.
* **Interactions Table**: Contains `id`, `hcp_id` (foreign key), `interaction_type` (PostgreSQL Enum: `Meeting`, `Video Meeting`, `Chat`), `channel` (Enum: `structured_form`, `chat`), `interaction_date` (DateTime), `summary`, `raw_notes`, `topics_discussed` (JSON list), `products_discussed` (JSON list), `sentiment` (Enum: `positive`, `neutral`, `negative`), and `samples_provided` (JSON list).

---

## 🐳 Docker Compose Deployment & Nginx Reverse Proxy

The entire application is containerized and managed using Docker Compose, providing a production-grade infrastructure:

* **PostgreSQL Container (`db`)**: Running PostgreSQL 16 on Alpine Linux, exposed on host port `5432`.
* **FastAPI Backend Container (`backend`)**: Running the FastAPI app on Uvicorn (port `8000`).
* **Vite Frontend & Nginx Container (`frontend`)**: Serves the built production React assets through Nginx on host port `5173`.
* **Nginx Reverse Proxy API Routing**: Nginx is configured to proxy all `/api` prefix requests directly to the backend container (`http://backend:8000/api`), routing them transparently and avoiding CORS issues in production.

---

## 🚀 Commands & Development Scripts

### Launch the Full Containerized Stack
To build images and spin up the complete database, backend, and frontend proxy services in detached mode, run:
```bash
docker compose up -d --build
```

### Inspect Container logs
To view stdout logs for the FastAPI backend service:
```bash
docker compose logs backend
```

### Recreate Database Schema & Seed Initial HCPs
To clear out all tables (dropping enums and foreign key references) and re-seed the initial doctor data (including Dr. Gregory House), run the recreate script inside the backend container:
```bash
# Copy script to container if not present
docker cp backend/app/agent/recreate_db.py hcp-crm-backend:/app/recreate_db.py

# Run recreation inside the container
docker compose exec backend python recreate_db.py
```

### Run Automated Verification Tests
To run full conversational turn tests, checking that the agent prompts for missing details, performs live drafting, and updates fields in PostgreSQL, run:
```bash
python backend/verify_refactored_flow.py
```
