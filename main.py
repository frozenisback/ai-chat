from flask import Flask, render_template_string, request, Response, session
import os, requests, json, time

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

INFERENCE_URL = os.environ.get("INFERENCE_URL")
INFERENCE_KEY = os.environ.get("INFERENCE_KEY")
INFERENCE_MODEL_ID = os.environ.get("INFERENCE_MODEL_ID")

# ---------------- HTML (Modern UI) ---------------- #
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Kust — Personal AI Chat</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link href="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/styles/github-dark.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/lib/common.min.js"></script>
<style>
body { background: #0d1117; color: #e6edf3; font-family: 'Inter', sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; }
.chat-container { flex: 1; overflow-y: auto; padding: 20px; }
.message { margin-bottom: 18px; line-height: 1.5; }
.user { text-align: right; }
.assistant code { background: #161b22; padding: 2px 5px; border-radius: 4px; }
pre { background: #161b22; padding: 10px; border-radius: 8px; position: relative; overflow-x: auto; }
.copy-btn { position: absolute; top: 8px; right: 8px; background: #21262d; border: none; color: #e6edf3; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 12px; }
.input-box { display: flex; padding: 15px; background: #161b22; border-top: 1px solid #30363d; }
input { flex: 1; background: #0d1117; border: 1px solid #30363d; color: white; border-radius: 8px; padding: 10px; }
button { margin-left: 10px; background: #238636; border: none; color: white; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-weight: 500; }
button:hover { background: #2ea043; }
.header { padding: 12px 20px; background: #161b22; border-bottom: 1px solid #30363d; font-weight: 600; }
</style>
</head>
<body>
<div class="header">Kust — Personal AI Chat</div>
<div id="chat" class="chat-container"></div>
<div class="input-box">
  <input id="prompt" placeholder="Ask anything..." autocomplete="off" />
  <button onclick="sendMsg()">Send</button>
</div>
<script>
const chat = document.getElementById('chat');
let controller;

function appendMessage(role, text, append=false) {
  let last = chat.lastElementChild;
  if (append && last && last.classList.contains(role)) {
    last.querySelector('.content').innerHTML = marked.parse(text);
    hljs.highlightAll();
    return;
  }
  const div = document.createElement('div');
  div.className = 'message ' + role;
  div.innerHTML = '<div class="content">' + marked.parse(text) + '</div>';
  chat.appendChild(div);
  hljs.highlightAll();
  div.querySelectorAll('pre code').forEach(block => {
    const btn = document.createElement('button');
    btn.innerText = 'Copy';
    btn.className = 'copy-btn';
    btn.onclick = () => navigator.clipboard.writeText(block.innerText);
    block.parentElement.appendChild(btn);
  });
  chat.scrollTop = chat.scrollHeight;
}

async function sendMsg() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;
  appendMessage('user', prompt);
  document.getElementById('prompt').value = '';
  appendMessage('assistant', '▌');
  const resp = await fetch('/chat', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({prompt}) });
  if (!resp.ok) { appendMessage('assistant', '[Error: Network issue]', true); return; }
  const reader = resp.body.getReader();
  let decoder = new TextDecoder(), buf = '';
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buf += decoder.decode(value, {stream:true});
    appendMessage('assistant', buf, true);
  }
}
</script>
</body>
</html>
"""

# ---------------- Memory ---------------- #
def get_chat_history():
    return session.get("history", [])

def add_to_history(role, content):
    history = session.get("history", [])
    history.append({"role": role, "content": content})
    session["history"] = history[-20:]  # limit memory

# ---------------- Routes ---------------- #
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_msg = data.get("prompt", "").strip()
    if not user_msg:
        return "Empty prompt", 400

    add_to_history("user", user_msg)
    messages = [{"role": "system", "content": "You are Kust's personal AI assistant. Be concise, helpful, and code-friendly."}]
    messages.extend(get_chat_history())

    def stream():
        headers = {
            "Authorization": f"Bearer {INFERENCE_KEY}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json"
        }
        payload = {"model": INFERENCE_MODEL_ID, "messages": messages, "stream": True}
        with requests.post(f"{INFERENCE_URL}/v1/chat/completions", headers=headers, json=payload, stream=True, timeout=120) as r:
            text_buf = ""
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"): 
                    continue
                if line.strip() == "data: [DONE]":
                    break
                try:
                    data = json.loads(line[5:])
                    delta = data["choices"][0]["delta"]
                    if "content" in delta:
                        chunk = delta["content"]
                        text_buf += chunk
                        yield chunk
                except Exception:
                    continue
            add_to_history("assistant", text_buf)
    return Response(stream(), mimetype="text/plain")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
