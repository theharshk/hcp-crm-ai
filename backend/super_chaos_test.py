import requests
import json
import sys
import time

url = 'http://localhost:8000/api/chat'
session_id = 'super-chaos-session'
history = []

draft_interaction = None

def send(msg, hcp_id=None):
    global history, draft_interaction
    payload = {
        'message': msg,
        'history': history,
        'session_id': session_id,
        'hcp_id': hcp_id,
        'draft_interaction': draft_interaction
    }
    time.sleep(3) # Groq limits
    r = requests.post(url, json=payload, timeout=120)
    if r.status_code != 200:
        print(f"Error: {r.status_code} - {r.text}")
        sys.exit(1)
    data = r.json()
    reply = data.get("reply")
    state = data.get("state") or {}
    
    draft_interaction = state.get("draft_interaction")
    
    history.append({'role': 'user', 'content': msg})
    history.append({'role': 'assistant', 'content': reply})
    
    print(f"USER: {msg}")
    print(f"AGENT: {reply}")
    state_hcp = state.get("hcp")
    hcp_id_resolved = state_hcp.get("id") if state_hcp else None
    print(f"HCP: {state_hcp.get('name') if state_hcp else 'None'} ({hcp_id_resolved})")
    print(f"DRAFT_INTERACTION: {json.dumps(state.get('draft_interaction'))}")
    print("-" * 50)
    
    return hcp_id_resolved, state

print("=== STARTING SUPER CHAOS WORKFLOW TEST ===")

# 1. Bypass python intercept with weird spacing/punctuation
hcp_id, state = send("Dr.   Gregory,,, House..")

# 2. Garbage inputs during extraction
hcp_id, state = send("log interaction", hcp_id)
hcp_id, state = send("ummm idk maybe a video call or maybe just an email but let's go with video call", hcp_id)

# 3. Rambling topic with weird punctuation
hcp_id, state = send("topics? we talked about lupus... wait no, cancer. actually just the weather. wait, definitely cancer. and some other stuff like... idk, immunology.", hcp_id)

# 4. Conflicting sentiment
hcp_id, state = send("he was super angry at first, then happy, but overall it was a terrible, negative experience, wait no it was positive.", hcp_id)

# 5. Overloading the arrays with weird quantities
hcp_id, state = send("I gave him 5 boxes of Vicodin, half a box of Tylenol, and wait... take away the Vicodin.", hcp_id)

# 6. Intent hijacking (mid-draft, ask to edit an old one)
hcp_id, state = send("actually can you edit my previous interaction instead?", hcp_id)

# 7. Complete nonsense
hcp_id, state = send("fjwioefjweiofjweiof", hcp_id)

# 8. Try to save with conflicting doctor name in the save command
hcp_id, state = send("save this for Dr Fake Tester", hcp_id)

print("=== TEST COMPLETE ===")
