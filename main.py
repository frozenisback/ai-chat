#!/usr/bin/env python3
"""
main.py - Single-file Flask chat app (clean UI, streaming tokens, code blocks with copy,
          per-session conversation memory). Uses Heroku Inference addon config vars:

  INFERENCE_URL, INFERENCE_KEY, INFERENCE_MODEL_ID

Save as main.py and run as usual. Keep in mind conversation memory is in-memory per dyno.
"""

from flask import Flask, request, render_template_string, jsonify, make_response, Response, stream_with_context
import os, requests, json, re, uuid, time, threading, html

# ---------- Config ----------
INFERENCE_URL = os.environ.get("INFERENCE_URL")
INFERENCE_KEY = os.environ.get("INFERENCE_KEY")
INFERENCE_MODEL_ID = os.environ.get("INFERENCE_MODEL_ID")

# Basic validation (app will still start; endpoints will error clearly)
if not INFERENCE_URL or not INFERENCE_KEY or not INFERENCE_MODEL_ID:
    print("WARNING: INFERENCE_URL, INFERENCE_KEY or INFERENCE_MODEL_ID not set. Set them in Heroku config vars.", flush=True)

# ---------- App ----------
app = Flask(__name__)

# In-memory conversation store: { session_id: [ {role:'user'|'assistant', 'content': '...'}, ... ] }
# Limited to recent N messages to avoid huge prompts
CHAT_STORE = {}
CHAT_LIMIT = 20

# ---------- Helpers ----------
CODE_FENCE_RE = re.compile(r"```(?:([\w+-]+)\n)?(.*?)```", re.S)

def extract_first_code_block(text: str):
    m = CODE_FENCE_RE.search(text or "")
    if m:
        lang = m.group(1) or None
        code = m.group(2)
        return lang, code
    return None, None

def session_id_from_request(req):
    sid = req.cookies.get("kust_sid")
    if not sid:
        sid = str(uuid.uuid4())
    return sid

def append_message(sid, role, content):
    lst = CHAT_STORE.setdefault(sid, [])
    lst.append({"role": role, "content": content})
    if len(lst) > CHAT_LIMIT:
        # keep last N
        CHAT_STORE[sid] = lst[-CHAT_LIMIT:]

def build_openai_messages(sid, user_message):
    """
    Build messages list for OpenAI-style chat completions.
    Uses conversation history from CHAT_STORE and appends the current user message.
    """
    msgs = []
    history = CHAT_STORE.get(sid, [])
    for m in history:
        # ensure role mapping
        role = m.get("role", "user")
        content = m.get("content", "")
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_message})
    return msgs

def post_json(url, headers, payload, timeout=30, stream=False):
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout, stream=stream)
        return r
    except Exception as e:
        return {"exception": str(e)}

# Robust non-streaming call (tries multiple shapes)
def call_ai_nonstream(sid, user_message, timeout=30):
    """
    Returns dict {"ok":True,"text":...} or {"error": "..."}
    Uses conversation history to provide context.
    """
    base = INFERENCE_URL.rstrip("/") if INFERENCE_URL else None
    headers = {"Authorization": f"Bearer {INFERENCE_KEY}", "Content-Type": "application/json"}

    # 1) Try OpenAI-style chat completions with messages
    try:
        chat_url = f"{base}/v1/chat/completions"
        payload = {"model": INFERENCE_MODEL_ID, "messages": build_openai_messages(sid, user_message)}
        r = post_json(chat_url, headers, payload, timeout=timeout)
        if isinstance(r, requests.Response) and r.status_code == 200:
            jr = r.json()
            # OpenAI-style: choices[0].message.content
            if isinstance(jr, dict):
                if "choices" in jr and jr["choices"]:
                    ch = jr["choices"][0]
                    msg = ch.get("message") or {}
                    content = msg.get("content") if isinstance(msg, dict) else None
                    if content:
                        return {"ok": True, "text": content}
                    # fallback to 'text'
                    if "text" in ch and isinstance(ch["text"], str):
                        return {"ok": True, "text": ch["text"]}
                # fallback to 'output'
                if "output" in jr:
                    out = jr["output"]
                    if isinstance(out, str):
                        return {"ok": True, "text": out}
                    elif isinstance(out, list):
                        # join
                        return {"ok": True, "text": "\n".join(map(str, out))}
            return {"ok": True, "text": r.text}
    except Exception:
        pass

    # 2) Try /v1/models/{model_id}/invoke with concatenated prompt
    try:
        invoke_url = f"{base}/v1/models/{INFERENCE_MODEL_ID}/invoke"
        # Build a single prompt that includes last messages (simple concatenation)
        history = CHAT_STORE.get(sid, [])
        history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history])
        full_prompt = f"{history_text}\nUSER: {user_message}"
        payload = {"input": full_prompt}
        r = post_json(invoke_url, headers, payload, timeout=timeout)
        if isinstance(r, requests.Response) and r.status_code == 200:
            jr = r.json()
            if isinstance(jr, dict):
                if "output" in jr:
                    return {"ok": True, "text": jr.get("output")}
                if "result" in jr:
                    return {"ok": True, "text": jr.get("result")}
            return {"ok": True, "text": r.text}
    except Exception:
        pass

    # 3) try /invoke
    try:
        base_invoke = f"{base}/invoke"
        payload = {"input": user_message}
        r = post_json(base_invoke, headers, payload, timeout=timeout)
        if isinstance(r, requests.Response) and r.status_code == 200:
            try:
                jr = r.json()
                if isinstance(jr, dict) and "output" in jr:
                    return {"ok": True, "text": jr.get("output")}
            except Exception:
                return {"ok": True, "text": r.text}
    except Exception:
        pass

    # 4) final fallback: POST to base
    try:
        r = post_json(base, headers, {"input": user_message}, timeout=timeout)
        if isinstance(r, requests.Response) and r.status_code == 200:
            try:
                jr = r.json()
                if isinstance(jr, dict) and "output" in jr:
                    return {"ok": True, "text": jr.get("output")}
            except Exception:
                return {"ok": True, "text": r.text}
        else:
            if isinstance(r, dict) and r.get("exception"):
                return {"error": f"Network error: {r.get('exception')}"}
    except Exception as e:
        return {"error": str(e)}

    return {"error": "All inference attempts failed or returned non-200 status."}

# Streaming generator: tries OpenAI-style streaming first, falls back to non-stream result
def stream_ai_generator(sid, user_message, timeout=60):
    """
    Yields successive text chunks (str) as they arrive.
    Tries OpenAI-style streaming first (chat completions stream).
    If that fails, yields the full non-stream text as one chunk.
    """
    base = INFERENCE_URL.rstrip("/") if INFERENCE_URL else None
    headers = {"Authorization": f"Bearer {INFERENCE_KEY}", "Content-Type": "application/json"}

    # Try OpenAI-style streaming
    try:
        chat_url = f"{base}/v1/chat/completions"
        payload = {"model": INFERENCE_MODEL_ID, "messages": build_openai_messages(sid, user_message), "stream": True}
        r = post_json(chat_url, headers, payload, timeout=timeout, stream=True)
        if isinstance(r, requests.Response) and r.status_code == 200:
            # Process SSE format
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                
                # Handle SSE format with "data:" prefix
                if line.startswith("data:"):
                    # Extract the actual message content after "data:"
                    data_content = line[len("data:"):].strip()
                    
                    # Skip empty messages
                    if not data_content:
                        continue
                    
                    # Check for [DONE] marker
                    if data_content == "[DONE]":
                        break
                    
                    # Try to parse as JSON
                    try:
                        # HTML decode the content first
                        decoded_content = html.unescape(data_content)
                        js = json.loads(decoded_content)
                        
                        # OpenAI-style: choices[0].delta.content
                        if "choices" in js and js["choices"]:
                            ch = js["choices"][0]
                            # delta
                            delta = ch.get("delta", {})
                            content = None
                            if isinstance(delta, dict):
                                content = delta.get("content")
                            # fallback for other shapes
                            if not content:
                                # maybe choices[0].message.content
                                msg = ch.get("message") or {}
                                content = msg.get("content") if isinstance(msg, dict) else None
                            if content:
                                yield content
                                continue
                        # other shapes: maybe 'output' text
                        if "output" in js:
                            out = js["output"]
                            if isinstance(out, str):
                                yield out
                            elif isinstance(out, list):
                                yield " ".join(map(str, out))
                            else:
                                yield json.dumps(out)
                            continue
                    except json.JSONDecodeError:
                        # Not valid JSON, treat as plain text
                        # HTML decode first
                        decoded_content = html.unescape(data_content)
                        yield decoded_content
            return
        # if non-200 or not a Response, fall through to non-stream
    except Exception as e:
        print(f"Streaming error: {e}")
        pass

    # Fallback: non-streaming call
    r = call_ai_nonstream(sid, user_message, timeout=timeout)
    if "ok" in r and r["ok"]:
        yield r.get("text", "")
    else:
        yield f"[Error contacting inference: {r.get('error','unknown')}]"

# ---------- HTML (clean modern chat UI) ----------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Kust — Clean AI Chat</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root{
      --bg:#0f1724; --card:#071023; --muted:#94a3b8; --accent:#7c3aed; --glass:rgba(255,255,255,0.03);
      --bubble-user:linear-gradient(180deg,#0b1f2b,#09303f);
      --bubble-bot:linear-gradient(180deg,#071829,#072a3a);
    }
    *{box-sizing:border-box}
    html,body{height:100%;margin:0;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;font-weight:500;}
    body{background:linear-gradient(180deg,#041322 0%, #06182a 100%); color:#e6eef8; display:flex; align-items:center; justify-content:center; padding:10px;}
    .app{width:100%;max-width:1400px;background:var(--card);border-radius:14px;box-shadow:0 10px 40px rgba(2,6,23,0.6);overflow:hidden;border:1px solid rgba(255,255,255,0.03); display:flex; flex-direction:column; height:95vh;}
    header{display:flex;align-items:center;justify-content:space-between;padding:18px 22px;border-bottom:1px solid rgba(255,255,255,0.02); flex-shrink:0;}
    header .title{display:flex;gap:12px;align-items:center}
    .logo{width:44px;height:44px;border-radius:10px;background:linear-gradient(135deg,#7c3aed,#06b6d4);display:flex;align-items:center;justify-content:center;font-weight:700}
    header h1{font-size:16px;margin:0;font-weight:600}
    header .meta{color:var(--muted);font-size:13px}
    .wrap{display:flex;gap:20px;padding:22px; flex:1; overflow:hidden;}
    .chat{flex:1;display:flex;flex-direction:column;min-width:0;}
    .messages{flex:1;overflow:auto;padding:18px;display:flex;flex-direction:column;gap:12px;}
    .msg{max-width:82%;padding:12px 14px;border-radius:12px;word-break:break-word}
    .msg.user{margin-left:auto;background:var(--bubble-user);border:1px solid rgba(255,255,255,0.03)}
    .msg.bot{margin-right:auto;background:var(--bubble-bot);border:1px solid rgba(255,255,255,0.02)}
    .meta-small{font-size:12px;color:var(--muted);margin-top:6px}
    .input-row{display:flex;gap:8px;padding:12px;align-items:center;border-top:1px solid rgba(255,255,255,0.02);background:linear-gradient(180deg, rgba(255,255,255,0.01), transparent); flex-shrink:0;}
    .input{flex:1;padding:12px;border-radius:10px;border:1px solid rgba(255,255,255,0.03);background:transparent;color:inherit;font-size:14px;}
    button.primary{background:var(--accent);border:none;padding:10px 14px;border-radius:10px;color:white;cursor:pointer;font-weight:600;}
    button.ghost{background:transparent;border:1px solid rgba(255,255,255,0.04);padding:9px 12px;border-radius:10px;color:var(--muted);cursor:pointer;}
    /* right panel */
    .panel{width:280px;border-left:1px solid rgba(255,255,255,0.02);padding-left:18px;display:flex;flex-direction:column;gap:12px; flex-shrink:0;}
    .panel .card{background:transparent;border-radius:8px;padding:10px;border:1px solid rgba(255,255,255,0.02);}
    .small{font-size:13px;color:var(--muted);}
    /* code blocks */
    pre{background:#0b1220;padding:12px;border-radius:8px;overflow:auto;border:1px solid rgba(255,255,255,0.02);white-space:pre-wrap;}
    .code-wrap{position:relative;}
    .copy-btn{position:absolute;right:8px;top:8px;background:rgba(255,255,255,0.04);border-radius:6px;padding:6px 8px;border:0;color:#cfe9ff;cursor:pointer;font-size:12px;}
    .streaming-cursor{display:inline-block;width:6px;height:12px;background:rgba(255,255,255,0.8);margin-left:6px;vertical-align:middle;border-radius:2px;animation: blink 1s linear infinite;}
    @keyframes blink{0%{opacity:1}50%{opacity:0.15}100%{opacity:1}}
    .stamp{font-size:12px;color:var(--muted);margin-top:6px;}
    
    /* Responsive design */
    @media (max-width: 1024px) {
      .panel { display: none; }
      .msg { max-width: 90%; }
    }
    
    @media (max-width: 768px) {
      body { padding: 5px; }
      .app { height: 98vh; border-radius: 8px; }
      header { padding: 12px 15px; }
      header h1 { font-size: 14px; }
      .wrap { padding: 15px; gap: 15px; }
      .messages { padding: 10px; }
      .input-row { padding: 10px; }
      .msg { max-width: 95%; padding: 10px 12px; }
      button.primary, button.ghost { padding: 8px 12px; font-size: 14px; }
    }
    
    @media (max-width: 480px) {
      header .title { gap: 8px; }
      .logo { width: 36px; height: 36px; font-size: 14px; }
      header h1 { font-size: 12px; }
      header .meta { font-size: 11px; }
      .wrap { padding: 10px; gap: 10px; }
      .messages { padding: 8px; gap: 8px; }
      .input-row { padding: 8px; gap: 5px; }
      .msg { max-width: 98%; padding: 8px 10px; font-size: 14px; }
      .input { padding: 10px; font-size: 14px; }
      button.primary, button.ghost { padding: 8px 10px; font-size: 13px; }
    }
  </style>
  <!-- highlightjs -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
</head>
<body>
  <div class="app" role="main">
    <header>
      <div class="title">
        <div class="logo">K</div>
        <div>
          <h1>Kust — Personal AI Chat</h1>
          <div class="meta">Model: <span id="modelid">loading...</span> • Private use</div>
        </div>
      </div>
    </header>

    <div class="wrap">
      <div class="chat">
        <div class="messages" id="messages" role="log" aria-live="polite"></div>

        <div class="input-row">
          <input id="inputBox" class="input" placeholder="Ask the AI anything — code, explainers, ideas..." />
          <button id="sendBtn" class="primary">Send</button>
        </div>
      </div>

      <aside class="panel">
        <div class="card">
          <div class="small"><strong>Tips</strong></div>
          <div class="small" style="margin-top:8px">• Ask the AI to write or explain code. <br>• Code blocks are copyable — click the copy button.</div>
        </div>

        <div class="card">
          <div class="small"><strong>Conversation</strong></div>
          <div style="margin-top:8px" class="small">This session remembers recent messages (in-memory on this server).</div>
        </div>

        <div class="card small">
          <div><strong>Model:</strong></div>
          <div id="modelInfo" class="stamp" style="margin-top:8px"></div>
        </div>
      </aside>
    </div>
  </div>

<script>
  const messagesEl = document.getElementById("messages");
  const inputBox = document.getElementById("inputBox");
  const sendBtn = document.getElementById("sendBtn");
  const modelIdEl = document.getElementById("modelid");

  // get model info and populate
  async function fetchModel() {
    try {
      const r = await fetch("/_model");
      const j = await r.json();
      modelIdEl.textContent = j.model || "unknown";
      document.getElementById("modelInfo").textContent = j.model ? j.model : "unknown";
    } catch (e) {
      modelIdEl.textContent = "unknown";
      console.error("Error fetching model:", e);
    }
  }
  fetchModel();

  // load history
  async function loadHistory() {
    try {
      const r = await fetch("/history");
      const j = await r.json();
      if (Array.isArray(j.history)) {
        j.history.forEach(m => {
          appendMessage(m.role, m.content, false);
        });
      }
    } catch (e) { 
      console.warn("Error loading history:", e);
    }
  }
  loadHistory();

  // helpers
  function appendMessage(role, htmlContent, scroll=true) {
    const d = document.createElement("div");
    d.className = "msg " + (role === "user" ? "user" : "bot");
    d.innerHTML = htmlContent;
    messagesEl.appendChild(d);
    processCodeBlocks(d); // attach copy buttons etc
    if (scroll) messagesEl.scrollTop = messagesEl.scrollHeight;
    return d;
  }

  function escapeHtml(s){ 
    return s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
  }

  // Convert AI reply text with code fences into HTML (pre/code)
  function formatReply(text, streaming=false) {
    if (!text) return "";
    
    // Track if we're inside a code block
    let inCodeBlock = false;
    let codeLanguage = "";
    let codeContent = "";
    let result = "";
    
    // Process the text character by character to handle streaming code blocks
    const lines = text.split('\\n');
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      
      // Check for code block start
      if (line.startsWith('```')) {
        if (!inCodeBlock) {
          // Starting a code block
          inCodeBlock = true;
          const parts = line.substring(3).trim();
          codeLanguage = parts || "text";
          codeContent = "";
        } else {
          // Ending a code block
          inCodeBlock = false;
          result += `<div class="code-wrap"><button class="copy-btn" onclick="copyCode(this)">Copy</button><pre><code class="language-${codeLanguage}">${escapeHtml(codeContent)}</code></pre></div>`;
          codeContent = "";
        }
      } else if (inCodeBlock) {
        // Inside a code block, collect content
        codeContent += (codeContent ? '\\n' : '') + line;
      } else {
        // Regular text, handle inline code and formatting
        let processedLine = line;
        
        // Handle inline code
        processedLine = processedLine.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Add the processed line
        result += processedLine + (i < lines.length - 1 ? '<br>' : '');
      }
    }
    
    // If we're still in a code block (streaming), add it
    if (inCodeBlock) {
      result += `<div class="code-wrap"><button class="copy-btn" onclick="copyCode(this)">Copy</button><pre><code class="language-${codeLanguage}">${escapeHtml(codeContent)}</code></pre></div>`;
    }
    
    return result;
  }

  // Attach copy buttons to code blocks (if created dynamically)
  function processCodeBlocks(container) {
    // highlight code
    container.querySelectorAll('pre code').forEach((el) => {
      hljs.highlightElement(el);
    });
    // copy buttons already in markup call copyCode()
  }

  window.copyCode = function(btn){
    try {
      const pre = btn.parentElement.querySelector('pre code');
      if (!pre) return;
      const txt = pre.innerText;
      navigator.clipboard.writeText(txt).then(()=> {
        btn.textContent = "Copied";
        setTimeout(()=> btn.textContent = "Copy", 1200);
      });
    } catch (e) {
      console.error(e);
    }
  };

  // Send (streaming)
  sendBtn.onclick = sendMessage;
  inputBox.addEventListener("keydown", (e)=>{ if(e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } });

  async function sendMessage(){
    const text = inputBox.value.trim();
    if (!text) return;
    inputBox.value = "";
    appendMessage("user", `<div><strong>You</strong><div class="meta-small">${escapeHtml(text)}</div></div>`);
    // create bot placeholder
    const botEl = appendMessage("bot", `<div class="meta-small">AI is typing...</div>`);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    // Attempt streaming endpoint
    try {
      const res = await fetch("/stream_chat", {
        method: "POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({message: text})
      });

      if (!res.ok) {
        // fallback to non-stream
        const j = await res.json();
        botEl.innerHTML = formatReply(j.reply || j.text || j.error || "No reply");
        processCodeBlocks(botEl);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let partial = "";
      // we'll replace the current botEl content as we stream
      botEl.innerHTML = `<div class="meta-small"><em>AI:</em> <span id="streaming_span"></span><span class="streaming-cursor"></span></div>`;
      const streamingSpan = botEl.querySelector("#streaming_span");

      while(true){
        const {done, value} = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, {stream:true});
        // Append chunk to partial and result
        partial += chunk;
        // Format and update the streaming content
        streamingSpan.innerHTML = formatReply(partial, true);
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
      // stream finished, remove cursor and finalize formatting
      streamingSpan.parentElement.innerHTML = formatReply(partial);
      processCodeBlocks(botEl);
      messagesEl.scrollTop = messagesEl.scrollHeight;

    } catch (err) {
      console.error("Streaming error:", err);
      // network error or streaming not supported: fallback to /chat
      try {
        const r = await fetch("/chat", {
          method:"POST", headers:{"Content-Type":"application/json"},
          body: JSON.stringify({message:text})
        });
        const j = await r.json();
        botEl.innerHTML = formatReply(j.reply || j.text || j.error || "No reply");
        processCodeBlocks(botEl);
        messagesEl.scrollTop = messagesEl.scrollHeight;
      } catch (e) {
        console.error("Fallback error:", e);
        botEl.innerHTML = `<div class="meta-small">Error: ${escapeHtml(String(e))}</div>`;
      }
    }
  }

  // On load, ensure code blocks have copy buttons etc
  document.addEventListener("click", (e)=>{
    // copy button fallback (if dynamically added)
    if (e.target && e.target.classList && e.target.classList.contains("copy-btn")) {
      e.preventDefault();
      copyCode(e.target);
    }
  });

</script>
</body>
</html>
"""

# ---------------------- Flask endpoints ----------------------

@app.route("/")
def index():
    # if no session cookie, set one
    sid = session_id_from_request(request)
    resp = make_response(render_template_string(INDEX_HTML))
    resp.set_cookie("kust_sid", sid, httponly=True, samesite="Lax")
    return resp

@app.route("/_model")
def modelinfo():
    return jsonify({"model": INFERENCE_MODEL_ID or ""})

@app.route("/history")
def history():
    sid = session_id_from_request(request)
    # ensure cookie set
    resp = make_response(jsonify({"history": CHAT_STORE.get(sid, [])}))
    resp.set_cookie("kust_sid", sid, httponly=True, samesite="Lax")
    return resp

@app.route("/clear", methods=["POST"])
def clear_chat():
    sid = session_id_from_request(request)
    CHAT_STORE.pop(sid, None)
    resp = make_response(jsonify({"ok": True}))
    resp.set_cookie("kust_sid", sid, httponly=True, samesite="Lax")
    return resp

@app.route("/chat", methods=["POST"])
def chat_nonstream_endpoint():
    """
    Non-streaming fallback: returns {"reply": "..."}
    Stores conversation in memory.
    """
    sid = session_id_from_request(request)
    data = request.get_json(force=True)
    user_msg = data.get("message", "")
    if not user_msg:
        return jsonify({"error": "No message provided"}), 400

    append_message(sid, "user", user_msg)
    r = call_ai_nonstream(sid, user_msg, timeout=30)
    if "error" in r:
        return jsonify({"error": r["error"]}), 500
    reply_text = r.get("text", "")
    append_message(sid, "assistant", reply_text)
    return jsonify({"reply": reply_text})

@app.route("/stream_chat", methods=["POST"])
def stream_chat_endpoint():
    """
    Streaming endpoint: returns a streaming response of text chunks.
    Also records conversation in memory (appends final reply).
    """
    sid = session_id_from_request(request)
    data = request.get_json(force=True)
    user_msg = data.get("message", "")
    if not user_msg:
        return jsonify({"error": "No message provided"}), 400

    append_message(sid, "user", user_msg)

    def generate():
        # stream generator yields bytes
        collected = []
        for chunk in stream_ai_generator(sid, user_msg, timeout=60):
            # chunk may be small text; yield it directly
            collected.append(chunk)
            try:
                yield chunk
            except GeneratorExit:
                break
        # store final aggregated response in history
        final = "".join(collected)
        append_message(sid, "assistant", final)

    # Return streamed response as plain text
    return Response(stream_with_context(generate()), content_type="text/plain; charset=utf-8")

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"Starting Kust chat on 0.0.0.0:{port} (debug={debug})", flush=True)
    app.run(host="0.0.0.0", port=port, debug=debug)
