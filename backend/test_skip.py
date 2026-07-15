import requests
import json
import time

url = "http://localhost:8000/api/chat"
session_id = "test-skip-session"
history = []
draft_interaction = None
hcp_id = "6515b1dc-e74e-4f95-ac6e-810ebe72a139" # Dr. Gregory House

def send(msg):
    global history, draft_interaction
    print(f"USER: {msg}")
    payload = {
        'message': msg,
        'history': history,
        'session_id': session_id,
        'hcp_id': hcp_id,
        'draft_interaction': draft_interaction
    }
    time.sleep(1)
    r = requests.post(url, json=payload, timeout=120)
    data = r.json()
    reply = data.get("reply")
    print(f"AGENT: {reply}")
    
    state = data.get("state") or {}
    draft_interaction = state.get("draft_interaction")
    print(f"DRAFT: {json.dumps(draft_interaction)}")
    print("-" * 50)
    
    history.append({'role': 'user', 'content': msg})
    history.append({'role': 'assistant', 'content': reply})

send("log interaction")
send("video call")
send("cancer and immunology")
send("positive")
send("No medications discussed")
send("[Skip Notes]")
send("[Skip Samples]")
