#!/usr/bin/env python3
"""
Single-file Flask app:

- Serves a modern chat UI
- Proxies chat to Heroku Inference (INFERENCE_URL, INFERENCE_KEY, INFERENCE_MODEL_ID)
- Tries multiple Heroku inference endpoint shapes to avoid 404s:
    1) /v1/chat/completions (OpenAI-compatible chat)
    2) /v1/models/{model_id}/invoke
    3) {INFERENCE_URL}/invoke
    4) direct POST to INFERENCE_URL with {"input": ...}
- Runs code snippets in an isolated temp dir with resource limits (POSIX)
- Offers an "auto-fix" endpoint that sends the failing code + error to the AI and returns suggested fixed code

WARNING:
    This app executes user-provided code on the host. Only run it locally or
    inside a VM/container for trusted use.
"""
from flask import Flask, render_template_string, request, jsonify
import os, requests, tempfile, subprocess, shutil, re, json, sys

# Optional resource limits (POSIX)
try:
    import resource
    POSIX = True
except Exception:
    POSIX = False

INFERENCE_URL = os.environ.get("INFERENCE_URL")
INFERENCE_KEY = os.environ.get("INFERENCE_KEY")
INFERENCE_MODEL_ID = os.environ.get("INFERENCE_MODEL_ID")

app = Flask(__name__, static_folder="static", template_folder="templates")

# -----------------------
# Helper: robust AI call
# -----------------------
def _post_json(url, headers, payload, timeout=30):
    """Helper wrapper for POST with exception handling."""
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        return r
    except Exception as e:
        return {"exception": str(e)}

def call_ai(prompt: str, timeout: int = 30):
    """
    Robust call to Heroku Inference endpoints.
    Tries multiple endpoint shapes and parses likely response formats.
    Returns dict: {"ok": True, "text": "..."} or {"error": "..."}
    """
    if not INFERENCE_URL or not INFERENCE_KEY:
        return {"error": "INFERENCE_URL or INFERENCE_KEY not configured on server."}

    base = INFERENCE_URL.rstrip("/")
    headers = {
        "Authorization": f"Bearer {INFERENCE_KEY}",
        "Content-Type": "application/json"
    }

    # 1) Try OpenAI-compatible chat completions
    try:
        chat_url = f"{base}/v1/chat/completions"
        payload = {"model": INFERENCE_MODEL_ID, "messages": [{"role":"user","content": prompt}]}
        r = _post_json(chat_url, headers, payload, timeout=timeout)
        if isinstance(r, requests.Response):
            if r.status_code == 200:
                try:
                    jr = r.json()
                    # Common shapes: jr['choices'][0]['message']['content'] or jr['output'] etc.
                    if isinstance(jr, dict):
                        # Try OpenAI-style choices
                        if "choices" in jr and isinstance(jr["choices"], list) and len(jr["choices"])>0:
                            ch = jr["choices"][0]
                            # message.content
                            msg = ch.get("message") or {}
                            content = msg.get("content") if isinstance(msg, dict) else None
                            if content:
                                return {"ok": True, "text": content}
                            # text or delta fallback
                            text = ch.get("text") or ch.get("message") or None
                            if isinstance(text, str):
                                return {"ok": True, "text": text}
                        # Heroku may respond with 'output'
                        if "output" in jr:
                            out = jr["output"]
                            if isinstance(out, str):
                                return {"ok": True, "text": out}
                            # sometimes output is dict
                            if isinstance(out, dict):
                                return {"ok": True, "text": out.get("text", str(out))}
                        # 'result' or 'data' fallback
                        if "result" in jr:
                            return {"ok": True, "text": jr["result"]}
                        if "data" in jr:
                            return {"ok": True, "text": jr["data"]}
                    # fallback to raw text
                    return {"ok": True, "text": r.text}
                except Exception as e:
                    return {"error": f"parsing chat response failed: {e} | status={r.status_code} body={getattr(r,'text',str(r))[:400]}"}
            else:
                # non-200 try next
                pass
        else:
            # network error returned
            return {"error": f"network error calling {chat_url}: {r.get('exception')}"}
    except Exception as e:
        # continue to next option
        pass

    # 2) Try model invoke: /v1/models/{id}/invoke
    try:
        invoke_url = f"{base}/v1/models/{INFERENCE_MODEL_ID}/invoke"
        payload = {"input": prompt}
        r = _post_json(invoke_url, headers, payload, timeout=timeout)
        if isinstance(r, requests.Response):
            if r.status_code == 200:
                try:
                    jr = r.json()
                    # Many Heroku model cards return {'output': "..."} or other shapes
                    if isinstance(jr, dict):
                        if "output" in jr:
                            return {"ok": True, "text": jr.get("output")}
                        if "result" in jr:
                            return {"ok": True, "text": jr.get("result")}
                        # fallback to any string value
                        for k in ("text","response","data"):
                            if k in jr and isinstance(jr[k], str):
                                return {"ok": True, "text": jr[k]}
                    return {"ok": True, "text": r.text}
                except Exception as e:
                    return {"error": f"parsing invoke response failed: {e} | status={r.status_code} body={getattr(r,'text',str(r))[:400]}"}
        else:
            return {"error": f"network error calling {invoke_url}: {r.get('exception')}"}
    except Exception:
        pass

    # 3) Try base /invoke
    try:
        base_invoke = f"{base}/invoke"
        payload = {"input": prompt}
        r = _post_json(base_invoke, headers, payload, timeout=timeout)
        if isinstance(r, requests.Response) and r.status_code == 200:
            try:
                jr = r.json()
                if isinstance(jr, dict) and "output" in jr:
                    return {"ok": True, "text": jr.get("output")}
                return {"ok": True, "text": r.text}
            except Exception:
                return {"ok": True, "text": r.text}
    except Exception:
        pass

    # 4) Last resort: POST to base url with {"input":...}
    try:
        r = _post_json(base, headers, {"input": prompt}, timeout=timeout)
        if isinstance(r, requests.Response) and r.status_code == 200:
            try:
                jr = r.json()
                if isinstance(jr, dict) and "output" in jr:
                    return {"ok": True, "text": jr.get("output")}
                return {"ok": True, "text": r.text}
            except Exception:
                return {"ok": True, "text": r.text}
        elif isinstance(r, dict) and r.get("exception"):
            return {"error": f"network error: {r.get('exception')}"}
        else:
            return {"error": f"All attempts failed. Last status: {getattr(r,'status_code', None)} body: {getattr(r,'text',str(r))[:400]}"}
    except Exception as e:
        return {"error": f"final fallback failed: {e}"}

# -----------------------
# Helper: extract code blocks
# -----------------------
CODE_FENCE_RE = re.compile(r"```(?:([\w+-]+)\n)?(.*?)```", re.S)

def extract_first_code_block(text: str):
    """
    Returns tuple(language, code) for the first triple-backtick block found.
    If none found, returns (None, text)
    """
    m = CODE_FENCE_RE.search(text or "")
    if m:
        lang = m.group(1) or None
        code = m.group(2)
        return lang, code
    return None, text

# -----------------------
# Helper: sandboxed run
# -----------------------
def _set_limits():
    # run in child process before exec (POSIX only)
    if not POSIX:
        return
    # CPU time seconds
    resource.setrlimit(resource.RLIMIT_CPU, (6, 8))   # 6s soft, 8s hard
    # address space (virtual memory) ~ bytes
    mem_bytes = 256 * 1024 * 1024  # 256MB
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    # file size
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))  # 10MB

def run_code(language: str, code: str, timeout_seconds: int = 10):
    """
    Runs code for 'python', 'node', or 'bash'.
    Returns dict with keys: success (bool), stdout, stderr, exit_code, timed_out (bool)
    """
    language = (language or "python").lower()
    tmpdir = tempfile.mkdtemp(prefix="ai_run_")
    filename = None
    cmd = None

    try:
        if language in ("py", "python"):
            filename = "main.py"
            with open(os.path.join(tmpdir, filename), "w", encoding="utf-8") as f:
                f.write(code)
            python_exec = shutil.which("python3") or shutil.which("python")
            if not python_exec:
                return {"success": False, "error": "python interpreter not found on server."}
            cmd = [python_exec, "-u", filename]

        elif language in ("js", "node", "javascript"):
            filename = "main.js"
            with open(os.path.join(tmpdir, filename), "w", encoding="utf-8") as f:
                f.write(code)
            node_exec = shutil.which("node")
            if not node_exec:
                return {"success": False, "error": "node not found on server."}
            cmd = [node_exec, filename]

        elif language in ("sh", "bash"):
            filename = "run.sh"
            with open(os.path.join(tmpdir, filename), "w", encoding="utf-8") as f:
                f.write(code)
            os.chmod(os.path.join(tmpdir, filename), 0o700)
            bash_exec = shutil.which("bash") or shutil.which("sh")
            if not bash_exec:
                return {"success": False, "error": "sh/bash not found on server."}
            cmd = [bash_exec, filename]

        else:
            return {"success": False, "error": f"Unsupported language: {language}"}

        proc = subprocess.run(
            cmd,
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            text=True,
            preexec_fn=_set_limits if POSIX else None
        )
        return {
            "success": proc.returncode == 0,
            "stdout": proc.stdout[:20000],
            "stderr": proc.stderr[:20000],
            "exit_code": proc.returncode,
            "timed_out": False,
        }

    except subprocess.TimeoutExpired as e:
        return {"success": False, "stdout": getattr(e, "output", "") or "", "stderr": getattr(e, "stderr", "") or "TIMEOUT", "exit_code": None, "timed_out": True}
    except Exception as e:
        return {"success": False, "error": f"Execution error: {e}"}
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

# -----------------------
# Flask routes (UI same as before)
# -----------------------

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Kust AI Chat & Code Runner</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <!-- Simple modern style -->
  <link href="https://cdn.jsdelivr.net/npm/modern-normalize/modern-normalize.css" rel="stylesheet">
  <style>
    :root{
      --bg:#0f1724; --card:#0b1220; --muted:#94a3b8; --accent:#7c3aed;
      --glass: rgba(255,255,255,0.03);
      --success:#10b981; --danger:#ef4444;
    }
    html,body{height:100%}
    body{
      background: linear-gradient(180deg,#071028 0%, #071427 100%);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
      color: #e6eef8; margin:0; padding:20px;
    }
    .wrap{max-width:1200px;margin:0 auto;display:grid;grid-template-columns: 1fr 420px;gap:18px;}
    .card{background:var(--card);border-radius:12px;padding:18px;box-shadow: 0 6px 24px rgba(2,6,23,0.6);border:1px solid rgba(255,255,255,0.02)}
    header{display:flex;align-items:center;gap:12px;margin-bottom:12px}
    header h1{font-size:18px;margin:0}
    .muted{color:var(--muted);font-size:13px}
    /* chat */
    .chat{height:72vh;display:flex;flex-direction:column;gap:12px}
    #messages{flex:1;overflow:auto;padding:8px;border-radius:8px;background:linear-gradient(180deg, rgba(255,255,255,0.01), transparent)}
    .msg{margin:8px 0; padding:10px;border-radius:8px;max-width:85%}
    .user{background:linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));align-self:flex-end;color:#bdebd6}
    .bot{background:linear-gradient(90deg, rgba(255,255,255,0.01), rgba(255,255,255,0.01));align-self:flex-start;color:#bfe0ff}
    .controls{display:flex;gap:8px;margin-top:8px}
    input[type="text"]{flex:1;padding:10px;border-radius:8px;border:1px solid rgba(255,255,255,0.03);background:transparent;color:inherit}
    button{padding:10px 14px;border-radius:8px;border:0;background:var(--accent);color:white;cursor:pointer}
    button.secondary{background:transparent;border:1px solid rgba(255,255,255,0.03)}
    /* right column */
    .panel{display:flex;flex-direction:column;gap:12px;height:72vh}
    textarea{width:100%;height:40%;resize:vertical;background:transparent;border:1px solid rgba(255,255,255,0.03);padding:10px;color:inherit;border-radius:8px}
    .log{flex:1;background:#03111b;padding:10px;border-radius:8px;overflow:auto;font-family:monospace;font-size:13px;color:#cfe9ff}
    select{padding:8px;border-radius:8px;background:transparent;border:1px solid rgba(255,255,255,0.03);color:inherit}
    small.note{color:var(--muted)}
    footer{margin-top:12px;text-align:center;color:var(--muted);font-size:12px}
    .pill{display:inline-block;padding:6px 8px;border-radius:999px;background:var(--glass);font-size:12px;border:1px solid rgba(255,255,255,0.02)}
    .row{display:flex;gap:8px}
  </style>
  <!-- highlight.js for code formatting -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <header>
        <div style="flex:1">
          <h1>Kust AI Chat — Personal</h1>
          <div class="muted">Private chat + code helper. AI model proxied through your Heroku Inference addon.</div>
        </div>
        <div style="text-align:right">
          <div class="pill">Model: <span id="modelid">unknown</span></div>
          <div style="height:6px"></div>
          <div class="muted">Local use only — do not expose publicly</div>
        </div>
      </header>

      <div class="chat">
        <div id="messages"></div>
        <div class="controls">
          <input id="user_input" type="text" placeholder="Ask the AI anything — tips: 'Write a python function to...'" />
          <button id="sendBtn">Send</button>
          <button id="fixBtn" class="secondary" title="Ask AI to suggest a fix for last error">Auto-fix</button>
        </div>
      </div>

      <footer><span class="muted">Powered by your Heroku Inference add-on • Kust Bots</span></footer>
    </div>

    <div class="card panel">
      <div>
        <div class="row" style="align-items:center">
          <select id="lang">
            <option value="python">Python</option>
            <option value="javascript">Node (JS)</option>
            <option value="bash">Bash</option>
          </select>
          <button id="runBtn">Run</button>
          <button id="clearBtn" class="secondary">Clear</button>
          <div style="flex:1"></div>
          <small class="note">Execution limits: 6s CPU, 256MB mem (POSIX). Logs truncated.</small>
        </div>
      </div>

      <textarea id="codeArea" spellcheck="false" placeholder="# paste code here or click an AI reply code block to load it"></textarea>

      <div class="log" id="logArea">Output and logs will appear here...</div>
    </div>
  </div>

<script>
  // minimal client logic
  const messagesEl = document.getElementById("messages");
  const inputEl = document.getElementById("user_input");
  const sendBtn = document.getElementById("sendBtn");
  const fixBtn = document.getElementById("fixBtn");
  const codeArea = document.getElementById("codeArea");
  const logArea = document.getElementById("logArea");
  const runBtn = document.getElementById("runBtn");
  const clearBtn = document.getElementById("clearBtn");
  const modelIdEl = document.getElementById("modelid");

  async function fetchModelInfo(){
    try {
      const r = await fetch("/_model");
      const j = await r.json();
      modelIdEl.textContent = j.model || "unknown";
    } catch (e) { /* ignore */ }
  }
  fetchModelInfo();

  function appendMsg(who, html){
    const d = document.createElement("div");
    d.className = "msg " + (who === "user" ? "user":"bot");
    d.innerHTML = html;
    messagesEl.appendChild(d);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    d.querySelectorAll && d.querySelectorAll("code").forEach(block=>{
      block.style.cursor = "pointer";
      block.addEventListener("click", ()=> {
        codeArea.value = block.innerText;
        log("Loaded code snippet into editor.");
      })
    });
    document.querySelectorAll('pre code').forEach((el) => { hljs.highlightElement(el); });
  }

  function log(text){
    const t = document.createElement("div");
    t.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
    logArea.appendChild(t);
    logArea.scrollTop = logArea.scrollHeight;
  }

  sendBtn.onclick = send;
  inputEl.addEventListener("keydown", (e)=>{ if(e.key === "Enter") send(); });

  async function send(){
    const text = inputEl.value.trim();
    if(!text) return;
    appendMsg("user", `<strong>You:</strong> ${escapeHtml(text)}`);
    inputEl.value = "";
    log("Sending to model...");
    try {
      const res = await fetch("/chat", {
        method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({message:text})
      });
      const j = await res.json();
      if(j.error) {
        appendMsg("bot", `<strong>AI Error:</strong> ${escapeHtml(j.error)}`);
        log("AI error: " + j.error);
      } else {
        let pretty = formatReply(j.reply || j.text || "");
        appendMsg("bot", pretty);
        log("AI replied.");
      }
    } catch (e) {
      appendMsg("bot", `<strong>Error:</strong> ${escapeHtml(String(e))}`);
      log("Network error: " + e);
    }
  }

  fixBtn.onclick = async ()=>{
    const code = codeArea.value;
    if(!code) { log("No code in editor to fix."); return; }
    const payload = { code, language: document.getElementById("lang").value, logs: logArea.innerText };
    appendMsg("user", `<strong>You:</strong> Auto-fix request`);
    log("Sending auto-fix request to AI...");
    try {
      const r = await fetch("/autofix", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)});
      const j = await r.json();
      if(j.error) {
        appendMsg("bot", `<strong>Auto-fix error:</strong> ${escapeHtml(j.error)}`);
        log("Auto-fix error: " + j.error);
      } else {
        const suggested = j.suggested || j.reply || "";
        appendMsg("bot", formatReply(suggested));
        const codeOnly = j.code || "";
        if(codeOnly) {
          codeArea.value = codeOnly;
          log("Loaded AI-suggested code into editor.");
        }
      }
    } catch (e) {
      appendMsg("bot", `<strong>Error:</strong> ${escapeHtml(String(e))}`);
      log("Auto-fix network error: " + e);
    }
  };

  runBtn.onclick = async ()=>{
    const language = document.getElementById("lang").value;
    const code = codeArea.value;
    if(!code) { log("No code to run."); return; }
    log("Running code...");
    try {
      const r = await fetch("/run", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({language, code})});
      const j = await r.json();
      if(j.error) {
        log("Run error: " + j.error);
      } else {
        log("Exit code: " + String(j.exit_code));
        if(j.timed_out) log("Timed out.");
        if(j.stdout) log("STDOUT:\\n" + j.stdout);
        if(j.stderr) log("STDERR:\\n" + j.stderr);
      }
    } catch (e) {
      log("Execution network error: " + e);
    }
  };

  clearBtn.onclick = ()=>{ codeArea.value = ""; logArea.innerText = ""; log("Cleared editor and logs."); };

  function escapeHtml(s){ return s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;'); }

  function formatReply(text) {
    if(!text) return "";
    const parts = text.split(/(```[\\s\\S]*?```)/g);
    return parts.map(p=>{
      if(p.startsWith("```")) {
        p = p.replace(/^```[\\w+\\-]*\\n?/, '').replace(/```$/, '');
        return `<pre><code>${escapeHtml(p)}</code></pre>`;
      } else {
        return `<div>${escapeHtml(p).replace(/\\n/g,'<br>')}</div>`;
      }
    }).join("");
  }
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/_model")
def modelinfo():
    model = INFERENCE_MODEL_ID or ""
    return jsonify({"model": model})

# Chat proxy (uses call_ai)
@app.route("/chat", methods=["POST"])
def chat_proxy():
    data = request.get_json(force=True)
    user_msg = data.get("message", "")
    if not user_msg:
        return jsonify({"error": "No message provided"}), 400

    # Build a helpful wrapper prompt for code-awareness
    prompt = (
        f"You are a helpful coding assistant. The user said:\n\n{user_msg}\n\n"
        "If you include code, wrap code blocks in triple backticks and label the language when possible.\n"
        "Respond succinctly but clearly."
    )

    r = call_ai(prompt, timeout=30)
    if "error" in r:
        return jsonify({"error": r["error"]}), 500
    reply = r.get("text") or ""
    return jsonify({"reply": reply})

# Run code
@app.route("/run", methods=["POST"])
def run_endpoint():
    data = request.get_json(force=True)
    language = data.get("language", "python")
    code = data.get("code", "")
    if not code:
        return jsonify({"error": "No code provided"}), 400
    res = run_code(language, code, timeout_seconds=10)
    return jsonify(res)

# Auto-fix endpoint
@app.route("/autofix", methods=["POST"])
def autofix_endpoint():
    data = request.get_json(force=True)
    code = data.get("code", "")
    logs = data.get("logs", "")
    language = data.get("language", "python")
    if not code:
        return jsonify({"error": "No code provided"}), 400

    prompt = (
        "You are an expert developer. The user provided the following code and logs.\n\n"
        "=== CODE ===\n"
        f"```{language}\n{code}\n```\n\n"
        "=== LOGS ===\n"
        f"{logs}\n\n"
        "Analyze the logs, find the bug(s), and provide a corrected version of the code. "
        "Return ONLY the corrected source file in triple backticks, labeled with the language, "
        "and nothing else. If multiple files are needed, return multiple fenced blocks with filenames as comments."
    )

    r = call_ai(prompt, timeout=60)
    if "error" in r:
        return jsonify({"error": r["error"]}), 500

    text = r.get("text", "")
    lang_found, code_block = extract_first_code_block(text)
    return jsonify({"suggested": text, "code": code_block or "", "lang": lang_found})

@app.route("/favicon.ico")
def favicon():
    return "", 204

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"Starting app on 0.0.0.0:{port} debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)
