#!/usr/bin/env python3
"""
Enhanced Cursor-like AI Code Editor and App Builder:

- Modern Cursor-like UI with file explorer, editor, terminal, and AI chat
- Project management with file system operations
- Package management (pip, npm, etc.)
- Enhanced code execution with project context
- AI assistance for code generation, debugging, and app building
- Workspace persistence
- Multi-language support with syntax highlighting
- Real-time collaboration features

WARNING:
    This app executes user-provided code on the host. Only run it locally or
    inside a VM/container for trusted use.
"""
from flask import Flask, render_template_string, request, jsonify, send_from_directory
import os, requests, tempfile, subprocess, shutil, re, json, sys, uuid, time, threading
from pathlib import Path
import signal

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

# Project storage
WORKSPACE_DIR = os.path.join(tempfile.gettempdir(), "cursor_workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# In-memory storage for active projects
active_projects = {}

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
    resource.setrlimit(resource.RLIMIT_CPU, (10, 12))   # 10s soft, 12s hard
    # address space (virtual memory) ~ bytes
    mem_bytes = 512 * 1024 * 1024  # 512MB
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    # file size
    resource.setrlimit(resource.RLIMIT_FSIZE, (20 * 1024 * 1024, 20 * 1024 * 1024))  # 20MB

def run_code(project_id: str, language: str, code: str, file_path: str = None, timeout_seconds: int = 15):
    """
    Runs code for 'python', 'node', or 'bash' in project context.
    Returns dict with keys: success (bool), stdout, stderr, exit_code, timed_out (bool)
    """
    language = (language or "python").lower()
    
    # Get project directory
    project_dir = os.path.join(WORKSPACE_DIR, project_id)
    os.makedirs(project_dir, exist_ok=True)
    
    # Determine file path
    if file_path:
        # Use provided file path
        file_dir = os.path.dirname(file_path)
        if file_dir:
            os.makedirs(os.path.join(project_dir, file_dir), exist_ok=True)
        filename = os.path.basename(file_path)
    else:
        # Use default filename based on language
        if language in ("py", "python"):
            filename = "main.py"
        elif language in ("js", "node", "javascript"):
            filename = "main.js"
        elif language in ("sh", "bash"):
            filename = "run.sh"
        else:
            filename = f"main.{language}"
    
    file_full_path = os.path.join(project_dir, filename)
    
    # Write code to file
    with open(file_full_path, "w", encoding="utf-8") as f:
        f.write(code)
    
    # Make executable if it's a script
    if language in ("sh", "bash"):
        os.chmod(file_full_path, 0o755)
    
    # Prepare command
    if language in ("py", "python"):
        python_exec = shutil.which("python3") or shutil.which("python")
        if not python_exec:
            return {"success": False, "error": "python interpreter not found on server."}
        cmd = [python_exec, "-u", filename]
    elif language in ("js", "node", "javascript"):
        node_exec = shutil.which("node")
        if not node_exec:
            return {"success": False, "error": "node not found on server."}
        cmd = [node_exec, filename]
    elif language in ("sh", "bash"):
        bash_exec = shutil.which("bash") or shutil.which("sh")
        if not bash_exec:
            return {"success": False, "error": "sh/bash not found on server."}
        cmd = [bash_exec, filename]
    else:
        return {"success": False, "error": f"Unsupported language: {language}"}

    try:
        proc = subprocess.run(
            cmd,
            cwd=project_dir,
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

# -----------------------
# Project Management
# -----------------------
def create_project(project_id: str, project_name: str = None):
    """Create a new project workspace"""
    project_dir = os.path.join(WORKSPACE_DIR, project_id)
    os.makedirs(project_dir, exist_ok=True)
    
    # Initialize project structure
    os.makedirs(os.path.join(project_dir, "src"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "public"), exist_ok=True)
    
    # Create package.json if it's a Node project
    if project_name and project_name.lower().endswith("js"):
        package_json = {
            "name": project_name,
            "version": "1.0.0",
            "description": "",
            "main": "src/index.js",
            "scripts": {
                "start": "node src/index.js",
                "dev": "nodemon src/index.js"
            },
            "dependencies": {},
            "devDependencies": {}
        }
        with open(os.path.join(project_dir, "package.json"), "w") as f:
            json.dump(package_json, f, indent=2)
    
    # Create requirements.txt if it's a Python project
    elif project_name and project_name.lower().endswith("py"):
        with open(os.path.join(project_dir, "requirements.txt"), "w") as f:
            f.write("# Add your requirements here\n")
    
    # Store project info
    active_projects[project_id] = {
        "name": project_name or f"Project {project_id[:8]}",
        "created_at": time.time(),
        "last_accessed": time.time()
    }
    
    return project_dir

def get_project_files(project_id: str, path: str = ""):
    """Get file tree for a project"""
    project_dir = os.path.join(WORKSPACE_DIR, project_id)
    if not os.path.exists(project_dir):
        return []
    
    target_dir = os.path.join(project_dir, path) if path else project_dir
    if not os.path.exists(target_dir):
        return []
    
    files = []
    try:
        for item in os.listdir(target_dir):
            item_path = os.path.join(target_dir, item)
            rel_path = os.path.relpath(item_path, project_dir)
            
            if os.path.isdir(item_path):
                files.append({
                    "name": item,
                    "path": rel_path,
                    "type": "directory",
                    "children": get_project_files(project_id, rel_path)
                })
            else:
                files.append({
                    "name": item,
                    "path": rel_path,
                    "type": "file"
                })
    except Exception as e:
        print(f"Error listing files: {e}")
    
    return sorted(files, key=lambda x: (x["type"] != "directory", x["name"].lower()))

def get_file_content(project_id: str, file_path: str):
    """Get content of a file"""
    project_dir = os.path.join(WORKSPACE_DIR, project_id)
    full_path = os.path.join(project_dir, file_path)
    
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        return None
    
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def save_file_content(project_id: str, file_path: str, content: str):
    """Save content to a file"""
    project_dir = os.path.join(WORKSPACE_DIR, project_id)
    full_path = os.path.join(project_dir, file_path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

def create_file(project_id: str, file_path: str, content: str = ""):
    """Create a new file"""
    project_dir = os.path.join(WORKSPACE_DIR, project_id)
    full_path = os.path.join(project_dir, file_path)
    
    if os.path.exists(full_path):
        return False, "File already exists"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True, "File created successfully"
    except Exception as e:
        return False, f"Error creating file: {e}"

def create_directory(project_id: str, dir_path: str):
    """Create a new directory"""
    project_dir = os.path.join(WORKSPACE_DIR, project_id)
    full_path = os.path.join(project_dir, dir_path)
    
    if os.path.exists(full_path):
        return False, "Directory already exists"
    
    try:
        os.makedirs(full_path)
        return True, "Directory created successfully"
    except Exception as e:
        return False, f"Error creating directory: {e}"

def delete_file_or_dir(project_id: str, path: str):
    """Delete a file or directory"""
    project_dir = os.path.join(WORKSPACE_DIR, project_id)
    full_path = os.path.join(project_dir, path)
    
    if not os.path.exists(full_path):
        return False, "Path does not exist"
    
    try:
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)
        return True, "Deleted successfully"
    except Exception as e:
        return False, f"Error deleting: {e}"

def install_dependencies(project_id: str, package_manager: str, packages: str):
    """Install dependencies for a project"""
    project_dir = os.path.join(WORKSPACE_DIR, project_id)
    
    if package_manager == "pip":
        # Install Python packages
        requirements_file = os.path.join(project_dir, "requirements.txt")
        
        # If packages are provided, add them to requirements.txt
        if packages:
            with open(requirements_file, "a") as f:
                f.write(f"\n{packages}")
        
        # Install using pip
        cmd = ["pip", "install", "-r", "requirements.txt"]
        
    elif package_manager == "npm":
        # Install Node packages
        if packages:
            cmd = ["npm", "install"] + packages.split()
        else:
            cmd = ["npm", "install"]
    
    elif package_manager == "yarn":
        # Install Node packages with yarn
        if packages:
            cmd = ["yarn", "add"] + packages.split()
        else:
            cmd = ["yarn", "install"]
    
    else:
        return {"success": False, "error": f"Unsupported package manager: {package_manager}"}
    
    try:
        proc = subprocess.run(
            cmd,
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60
        )
        
        return {
            "success": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Installation timed out"}
    except Exception as e:
        return {"success": False, "error": f"Installation error: {e}"}

# -----------------------
# Flask routes
# -----------------------

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Cursor AI - Code Editor & App Builder</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <!-- Monaco Editor -->
  <link rel="stylesheet" data-name="vs/editor/editor.main" href="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.34.1/min/vs/editor/editor.main.min.css">
  <!-- Font Awesome for icons -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    :root {
      --bg: #1e1e1e;
      --sidebar: #252526;
      --editor: #1e1e1e;
      --terminal: #1e1e1e;
      --accent: #007acc;
      --accent-hover: #1a85ff;
      --text: #cccccc;
      --text-dim: #969696;
      --border: #3e3e42;
      --selection: #264f78;
      --success: #4ec9b0;
      --error: #f48771;
      --warning: #dcdcaa;
    }

    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    html, body {
      height: 100%;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      color: var(--text);
      background-color: var(--bg);
      overflow: hidden;
    }

    .app-container {
      display: flex;
      flex-direction: column;
      height: 100vh;
    }

    .title-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 30px;
      background-color: var(--sidebar);
      padding: 0 10px;
      -webkit-app-region: drag;
      user-select: none;
    }

    .title-bar-left {
      display: flex;
      align-items: center;
    }

    .title-bar-right {
      display: flex;
      align-items: center;
    }

    .title-bar button {
      background: none;
      border: none;
      color: var(--text);
      padding: 0 8px;
      cursor: pointer;
      height: 100%;
      -webkit-app-region: no-drag;
    }

    .title-bar button:hover {
      background-color: rgba(255, 255, 255, 0.1);
    }

    .main-container {
      display: flex;
      flex: 1;
      overflow: hidden;
    }

    .sidebar {
      width: 50px;
      background-color: var(--sidebar);
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 10px 0;
    }

    .sidebar-item {
      width: 40px;
      height: 40px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 5px 0;
      border-radius: 5px;
      cursor: pointer;
      color: var(--text-dim);
      position: relative;
    }

    .sidebar-item:hover {
      background-color: rgba(255, 255, 255, 0.1);
      color: var(--text);
    }

    .sidebar-item.active {
      background-color: var(--accent);
      color: white;
    }

    .sidebar-item .tooltip {
      position: absolute;
      left: 50px;
      background-color: #333;
      color: white;
      padding: 5px 10px;
      border-radius: 3px;
      white-space: nowrap;
      opacity: 0;
      pointer-events: none;
      z-index: 1000;
    }

    .sidebar-item:hover .tooltip {
      opacity: 1;
    }

    .content-area {
      display: flex;
      flex: 1;
      overflow: hidden;
    }

    .panel {
      background-color: var(--sidebar);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .panel-header {
      height: 35px;
      display: flex;
      align-items: center;
      padding: 0 10px;
      background-color: rgba(0, 0, 0, 0.2);
      border-bottom: 1px solid var(--border);
    }

    .panel-content {
      flex: 1;
      overflow: auto;
    }

    .file-explorer {
      width: 220px;
      border-right: 1px solid var(--border);
    }

    .file-tree {
      padding: 5px 0;
    }

    .file-item {
      display: flex;
      align-items: center;
      padding: 3px 10px;
      cursor: pointer;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .file-item:hover {
      background-color: rgba(255, 255, 255, 0.1);
    }

    .file-item.active {
      background-color: var(--selection);
    }

    .file-item i {
      margin-right: 8px;
      width: 16px;
      text-align: center;
      font-size: 14px;
    }

    .file-item.folder i::before {
      content: "\\f07b";
    }

    .file-item.folder.expanded i::before {
      content: "\\f07c";
    }

    .file-item.python i::before {
      content: "\\e606";
      color: var(--warning);
    }

    .file-item.javascript i::before {
      content: "\\e60e";
      color: var(--warning);
    }

    .file-item.html i::before {
      content: "\\e60f";
      color: var(--error);
    }

    .file-item.css i::before {
      content: "\\e60d";
      color: var(--accent);
    }

    .file-item.json i::before {
      content: "\\e60b";
      color: var(--text-dim);
    }

    .file-item.txt i::before {
      content: "\\e60a";
      color: var(--text-dim);
    }

    .file-item.default i::before {
      content: "\\f15b";
      color: var(--text-dim);
    }

    .editor-container {
      flex: 1;
      display: flex;
      flex-direction: column;
    }

    .tabs-container {
      height: 35px;
      display: flex;
      background-color: var(--sidebar);
      overflow-x: auto;
      scrollbar-width: none;
    }

    .tabs-container::-webkit-scrollbar {
      display: none;
    }

    .tab {
      display: flex;
      align-items: center;
      padding: 0 10px;
      background-color: var(--editor);
      border-right: 1px solid var(--border);
      cursor: pointer;
      white-space: nowrap;
      min-width: 120px;
    }

    .tab:hover {
      background-color: rgba(255, 255, 255, 0.05);
    }

    .tab.active {
      background-color: var(--editor);
      border-bottom: 2px solid var(--accent);
    }

    .tab-name {
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .tab-close {
      margin-left: 8px;
      opacity: 0.7;
    }

    .tab-close:hover {
      opacity: 1;
    }

    .editor {
      flex: 1;
      overflow: hidden;
    }

    .bottom-panel {
      height: 200px;
      border-top: 1px solid var(--border);
      display: flex;
      flex-direction: column;
    }

    .panel-tabs {
      display: flex;
      background-color: var(--sidebar);
    }

    .panel-tab {
      padding: 5px 15px;
      cursor: pointer;
      border-bottom: 2px solid transparent;
    }

    .panel-tab:hover {
      background-color: rgba(255, 255, 255, 0.05);
    }

    .panel-tab.active {
      border-bottom: 2px solid var(--accent);
    }

    .terminal {
      flex: 1;
      background-color: var(--terminal);
      padding: 10px;
      font-family: 'Consolas', 'Monaco', monospace;
      font-size: 14px;
      overflow: auto;
      white-space: pre-wrap;
    }

    .ai-chat {
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 10px;
      overflow: hidden;
    }

    .chat-messages {
      flex: 1;
      overflow-y: auto;
      margin-bottom: 10px;
    }

    .chat-message {
      margin-bottom: 15px;
      display: flex;
    }

    .chat-message.user {
      justify-content: flex-end;
    }

    .chat-message-content {
      max-width: 80%;
      padding: 10px;
      border-radius: 5px;
    }

    .chat-message.user .chat-message-content {
      background-color: var(--accent);
      color: white;
    }

    .chat-message.assistant .chat-message-content {
      background-color: var(--sidebar);
    }

    .chat-input-container {
      display: flex;
    }

    .chat-input {
      flex: 1;
      background-color: var(--sidebar);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px;
      border-radius: 3px;
      resize: none;
    }

    .chat-send {
      margin-left: 10px;
      background-color: var(--accent);
      color: white;
      border: none;
      padding: 0 15px;
      border-radius: 3px;
      cursor: pointer;
    }

    .chat-send:hover {
      background-color: var(--accent-hover);
    }

    .context-menu {
      position: absolute;
      background-color: var(--sidebar);
      border: 1px solid var(--border);
      border-radius: 3px;
      padding: 5px 0;
      min-width: 150px;
      z-index: 1000;
      display: none;
    }

    .context-menu-item {
      padding: 5px 15px;
      cursor: pointer;
    }

    .context-menu-item:hover {
      background-color: var(--selection);
    }

    .context-menu-divider {
      height: 1px;
      background-color: var(--border);
      margin: 5px 0;
    }

    .modal {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-color: rgba(0, 0, 0, 0.5);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 2000;
    }

    .modal-content {
      background-color: var(--sidebar);
      border-radius: 5px;
      padding: 20px;
      width: 500px;
      max-width: 90%;
    }

    .modal-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 15px;
    }

    .modal-title {
      font-size: 18px;
      font-weight: 500;
    }

    .modal-close {
      background: none;
      border: none;
      color: var(--text);
      font-size: 20px;
      cursor: pointer;
    }

    .modal-body {
      margin-bottom: 15px;
    }

    .form-group {
      margin-bottom: 15px;
    }

    .form-label {
      display: block;
      margin-bottom: 5px;
    }

    .form-input {
      width: 100%;
      background-color: var(--editor);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px;
      border-radius: 3px;
    }

    .modal-footer {
      display: flex;
      justify-content: flex-end;
    }

    .btn {
      padding: 8px 15px;
      border-radius: 3px;
      cursor: pointer;
      border: none;
    }

    .btn-primary {
      background-color: var(--accent);
      color: white;
    }

    .btn-primary:hover {
      background-color: var(--accent-hover);
    }

    .btn-secondary {
      background-color: var(--border);
      color: var(--text);
    }

    .btn-secondary:hover {
      background-color: rgba(255, 255, 255, 0.1);
    }

    .status-bar {
      height: 22px;
      display: flex;
      align-items: center;
      padding: 0 10px;
      background-color: var(--accent);
      color: white;
      font-size: 12px;
    }

    .status-item {
      margin-right: 15px;
    }

    .notification {
      position: fixed;
      bottom: 30px;
      right: 20px;
      background-color: var(--sidebar);
      border: 1px solid var(--border);
      border-radius: 3px;
      padding: 10px 15px;
      min-width: 250px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
      z-index: 3000;
      transform: translateY(100px);
      opacity: 0;
      transition: transform 0.3s, opacity 0.3s;
    }

    .notification.show {
      transform: translateY(0);
      opacity: 1;
    }

    .notification.success {
      border-left: 4px solid var(--success);
    }

    .notification.error {
      border-left: 4px solid var(--error);
    }

    .notification.warning {
      border-left: 4px solid var(--warning);
    }

    .notification-title {
      font-weight: 500;
      margin-bottom: 5px;
    }

    .notification-message {
      font-size: 14px;
    }

    .hidden {
      display: none !important;
    }
  </style>
</head>
<body>
  <div class="app-container">
    <div class="title-bar">
      <div class="title-bar-left">
        <span>Cursor AI - Code Editor & App Builder</span>
      </div>
      <div class="title-bar-right">
        <button id="minimize-btn"><i class="fas fa-minus"></i></button>
        <button id="maximize-btn"><i class="fas fa-square"></i></button>
        <button id="close-btn"><i class="fas fa-times"></i></button>
      </div>
    </div>

    <div class="main-container">
      <div class="sidebar">
        <div class="sidebar-item active" data-panel="explorer">
          <i class="fas fa-folder"></i>
          <span class="tooltip">Explorer</span>
        </div>
        <div class="sidebar-item" data-panel="search">
          <i class="fas fa-search"></i>
          <span class="tooltip">Search</span>
        </div>
        <div class="sidebar-item" data-panel="git">
          <i class="fas fa-code-branch"></i>
          <span class="tooltip">Source Control</span>
        </div>
        <div class="sidebar-item" data-panel="debug">
          <i class="fas fa-bug"></i>
          <span class="tooltip">Run and Debug</span>
        </div>
        <div class="sidebar-item" data-panel="extensions">
          <i class="fas fa-puzzle-piece"></i>
          <span class="tooltip">Extensions</span>
        </div>
      </div>

      <div class="content-area">
        <div class="panel file-explorer" id="explorer-panel">
          <div class="panel-header">
            <span>EXPLORER</span>
            <div style="margin-left: auto;">
              <button id="new-file-btn" title="New File"><i class="fas fa-file-plus"></i></button>
              <button id="new-folder-btn" title="New Folder"><i class="fas fa-folder-plus"></i></button>
              <button id="refresh-btn" title="Refresh"><i class="fas fa-sync-alt"></i></button>
              <button id="collapse-btn" title="Collapse All"><i class="fas fa-compress-alt"></i></button>
            </div>
          </div>
          <div class="panel-content">
            <div class="file-tree" id="file-tree">
              <!-- File tree will be populated here -->
            </div>
          </div>
        </div>

        <div class="panel file-explorer hidden" id="search-panel">
          <div class="panel-header">
            <span>SEARCH</span>
          </div>
          <div class="panel-content">
            <div style="padding: 10px;">
              <input type="text" id="search-input" placeholder="Search" class="form-input">
              <div style="margin-top: 10px;">
                <input type="text" id="replace-input" placeholder="Replace" class="form-input">
              </div>
              <div style="margin-top: 10px;">
                <label><input type="checkbox" id="case-sensitive"> Match Case</label>
                <label style="margin-left: 10px;"><input type="checkbox" id="whole-word"> Whole Word</label>
                <label style="margin-left: 10px;"><input type="checkbox" id="regex"> Use Regular Expression</label>
              </div>
              <div style="margin-top: 10px;">
                <button class="btn btn-primary" id="search-btn">Search</button>
              </div>
            </div>
            <div id="search-results" style="padding: 10px;">
              <!-- Search results will be displayed here -->
            </div>
          </div>
        </div>

        <div class="panel file-explorer hidden" id="git-panel">
          <div class="panel-header">
            <span>SOURCE CONTROL</span>
          </div>
          <div class="panel-content">
            <div style="padding: 10px;">
              <p>Git functionality would be implemented here</p>
            </div>
          </div>
        </div>

        <div class="panel file-explorer hidden" id="debug-panel">
          <div class="panel-header">
            <span>RUN AND DEBUG</span>
          </div>
          <div class="panel-content">
            <div style="padding: 10px;">
              <div class="form-group">
                <label class="form-label">Select Language:</label>
                <select id="debug-language" class="form-input">
                  <option value="python">Python</option>
                  <option value="javascript">JavaScript (Node)</option>
                  <option value="bash">Bash</option>
                </select>
              </div>
              <div class="form-group">
                <button class="btn btn-primary" id="run-code-btn">Run Code</button>
                <button class="btn btn-secondary" id="debug-code-btn">Debug Code</button>
              </div>
              <div class="form-group">
                <label class="form-label">Package Management:</label>
                <div style="display: flex; margin-top: 5px;">
                  <select id="package-manager" class="form-input" style="margin-right: 10px;">
                    <option value="pip">pip</option>
                    <option value="npm">npm</option>
                    <option value="yarn">yarn</option>
                  </select>
                  <input type="text" id="package-name" placeholder="Package name(s)" class="form-input" style="flex: 1; margin-right: 10px;">
                  <button class="btn btn-primary" id="install-btn">Install</button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="panel file-explorer hidden" id="extensions-panel">
          <div class="panel-header">
            <span>EXTENSIONS</span>
          </div>
          <div class="panel-content">
            <div style="padding: 10px;">
              <p>Extensions would be listed here</p>
            </div>
          </div>
        </div>

        <div class="editor-container">
          <div class="tabs-container" id="tabs-container">
            <!-- Tabs will be added here -->
          </div>
          <div class="editor" id="editor">
            <!-- Monaco editor will be initialized here -->
          </div>
        </div>
      </div>
    </div>

    <div class="bottom-panel">
      <div class="panel-tabs">
        <div class="panel-tab active" data-panel="terminal">TERMINAL</div>
        <div class="panel-tab" data-panel="problems">PROBLEMS</div>
        <div class="panel-tab" data-panel="output">OUTPUT</div>
        <div class="panel-tab" data-panel="ai-chat">AI CHAT</div>
      </div>
      <div class="terminal" id="terminal-panel">
        <div id="terminal-content">
          <div>Welcome to Cursor AI Terminal</div>
          <div>Type 'help' for available commands</div>
          <div>$ </div>
        </div>
      </div>
      <div class="terminal hidden" id="problems-panel">
        <div>No problems detected</div>
      </div>
      <div class="terminal hidden" id="output-panel">
        <div id="output-content">
          <!-- Output will be displayed here -->
        </div>
      </div>
      <div class="ai-chat hidden" id="ai-chat-panel">
        <div class="chat-messages" id="chat-messages">
          <div class="chat-message assistant">
            <div class="chat-message-content">
              Hello! I'm your AI assistant. I can help you with coding, debugging, and building applications. How can I assist you today?
            </div>
          </div>
        </div>
        <div class="chat-input-container">
          <textarea id="chat-input" class="chat-input" placeholder="Ask me anything..."></textarea>
          <button id="chat-send" class="chat-send">Send</button>
        </div>
      </div>
    </div>

    <div class="status-bar">
      <div class="status-item">Ready</div>
      <div class="status-item" id="cursor-position">Ln 1, Col 1</div>
      <div class="status-item" id="file-type">Plain Text</div>
      <div class="status-item">UTF-8</div>
      <div class="status-item">Project: <span id="project-name">Untitled</span></div>
    </div>
  </div>

  <!-- Context Menu -->
  <div class="context-menu" id="context-menu">
    <div class="context-menu-item" id="ctx-open">Open</div>
    <div class="context-menu-item" id="ctx-rename">Rename</div>
    <div class="context-menu-item" id="ctx-delete">Delete</div>
    <div class="context-menu-divider"></div>
    <div class="context-menu-item" id="ctx-copy">Copy Path</div>
    <div class="context-menu-item" id="ctx-copy-rel">Copy Relative Path</div>
  </div>

  <!-- New File Modal -->
  <div class="modal hidden" id="new-file-modal">
    <div class="modal-content">
      <div class="modal-header">
        <div class="modal-title">New File</div>
        <button class="modal-close" id="new-file-close">&times;</button>
      </div>
      <div class="modal-body">
        <div class="form-group">
          <label class="form-label">File Name:</label>
          <input type="text" id="new-file-name" class="form-input" placeholder="example.py">
        </div>
        <div class="form-group">
          <label class="form-label">File Path:</label>
          <input type="text" id="new-file-path" class="form-input" placeholder="src/">
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn btn-secondary" id="new-file-cancel">Cancel</button>
        <button class="btn btn-primary" id="new-file-create">Create</button>
      </div>
    </div>
  </div>

  <!-- New Folder Modal -->
  <div class="modal hidden" id="new-folder-modal">
    <div class="modal-content">
      <div class="modal-header">
        <div class="modal-title">New Folder</div>
        <button class="modal-close" id="new-folder-close">&times;</button>
      </div>
      <div class="modal-body">
        <div class="form-group">
          <label class="form-label">Folder Name:</label>
          <input type="text" id="new-folder-name" class="form-input" placeholder="components">
        </div>
        <div class="form-group">
          <label class="form-label">Parent Path:</label>
          <input type="text" id="new-folder-path" class="form-input" placeholder="src/">
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn btn-secondary" id="new-folder-cancel">Cancel</button>
        <button class="btn btn-primary" id="new-folder-create">Create</button>
      </div>
    </div>
  </div>

  <!-- Notification -->
  <div class="notification" id="notification">
    <div class="notification-title" id="notification-title"></div>
    <div class="notification-message" id="notification-message"></div>
  </div>

  <!-- Monaco Editor -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.34.1/min/vs/loader.js"></script>
  <script>
    // Initialize Monaco Editor
    require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.34.1/min/vs' }});
    require(['vs/editor/editor.main'], function() {
      // Global variables
      let editor;
      let currentProjectId = 'default';
      let openTabs = [];
      let activeTab = null;
      let contextMenuTarget = null;

      // Initialize editor
      editor = monaco.editor.create(document.getElementById('editor'), {
        value: '// Welcome to Cursor AI\\n// Start coding or ask the AI assistant for help\\n',
        language: 'javascript',
        theme: 'vs-dark',
        automaticLayout: true,
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        fontSize: 14,
        tabSize: 2,
        wordWrap: 'on'
      });

      // Update cursor position
      editor.onDidChangeCursorPosition((e) => {
        const position = e.position;
        document.getElementById('cursor-position').textContent = `Ln ${position.lineNumber}, Col ${position.column}`;
      });

      // Initialize project
      initializeProject();

      // Setup event listeners
      setupEventListeners();

      // Load file tree
      loadFileTree();

      // Initialize functions
      function initializeProject() {
        fetch('/api/project/init', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ project_name: 'Untitled Project' })
        })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            currentProjectId = data.project_id;
            document.getElementById('project-name').textContent = data.project_name;
          }
        })
        .catch(error => {
          console.error('Error initializing project:', error);
        });
      }

      function setupEventListeners() {
        // Sidebar items
        document.querySelectorAll('.sidebar-item').forEach(item => {
          item.addEventListener('click', function() {
            // Remove active class from all items
            document.querySelectorAll('.sidebar-item').forEach(i => i.classList.remove('active'));
            // Add active class to clicked item
            this.classList.add('active');
            
            // Hide all panels
            document.querySelectorAll('.panel').forEach(panel => panel.classList.add('hidden'));
            
            // Show selected panel
            const panelId = this.dataset.panel + '-panel';
            const panel = document.getElementById(panelId);
            if (panel) panel.classList.remove('hidden');
          });
        });

        // Bottom panel tabs
        document.querySelectorAll('.panel-tab').forEach(tab => {
          tab.addEventListener('click', function() {
            // Remove active class from all tabs
            document.querySelectorAll('.panel-tab').forEach(t => t.classList.remove('active'));
            // Add active class to clicked tab
            this.classList.add('active');
            
            // Hide all panels
            document.querySelectorAll('.terminal, .ai-chat').forEach(panel => panel.classList.add('hidden'));
            
            // Show selected panel
            const panelId = this.dataset.panel + '-panel';
            const panel = document.getElementById(panelId);
            if (panel) panel.classList.remove('hidden');
          });
        });

        // New file button
        document.getElementById('new-file-btn').addEventListener('click', () => {
          document.getElementById('new-file-modal').classList.remove('hidden');
          document.getElementById('new-file-name').focus();
        });

        // New folder button
        document.getElementById('new-folder-btn').addEventListener('click', () => {
          document.getElementById('new-folder-modal').classList.remove('hidden');
          document.getElementById('new-folder-name').focus();
        });

        // Refresh button
        document.getElementById('refresh-btn').addEventListener('click', loadFileTree);

        // New file modal
        document.getElementById('new-file-create').addEventListener('click', createNewFile);
        document.getElementById('new-file-cancel').addEventListener('click', () => {
          document.getElementById('new-file-modal').classList.add('hidden');
        });
        document.getElementById('new-file-close').addEventListener('click', () => {
          document.getElementById('new-file-modal').classList.add('hidden');
        });

        // New folder modal
        document.getElementById('new-folder-create').addEventListener('click', createNewFolder);
        document.getElementById('new-folder-cancel').addEventListener('click', () => {
          document.getElementById('new-folder-modal').classList.add('hidden');
        });
        document.getElementById('new-folder-close').addEventListener('click', () => {
          document.getElementById('new-folder-modal').classList.add('hidden');
        });

        // Run code button
        document.getElementById('run-code-btn').addEventListener('click', runCode);

        // Install package button
        document.getElementById('install-btn').addEventListener('click', installPackage);

        // Chat send button
        document.getElementById('chat-send').addEventListener('click', sendChatMessage);
        document.getElementById('chat-input').addEventListener('keydown', (e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
          }
        });

        // Context menu
        document.addEventListener('contextmenu', (e) => {
          if (e.target.closest('.file-item')) {
            e.preventDefault();
            contextMenuTarget = e.target.closest('.file-item');
            showContextMenu(e.clientX, e.clientY);
          } else {
            hideContextMenu();
          }
        });

        document.addEventListener('click', () => {
          hideContextMenu();
        });

        // Context menu items
        document.getElementById('ctx-open').addEventListener('click', () => {
          if (contextMenuTarget) {
            openFile(contextMenuTarget.dataset.path);
          }
          hideContextMenu();
        });

        document.getElementById('ctx-rename').addEventListener('click', () => {
          if (contextMenuTarget) {
            const newName = prompt('Enter new name:', contextMenuTarget.textContent.trim());
            if (newName) {
              renameFile(contextMenuTarget.dataset.path, newName);
            }
          }
          hideContextMenu();
        });

        document.getElementById('ctx-delete').addEventListener('click', () => {
          if (contextMenuTarget && confirm(`Are you sure you want to delete ${contextMenuTarget.textContent.trim()}?`)) {
            deleteFile(contextMenuTarget.dataset.path);
          }
          hideContextMenu();
        });

        document.getElementById('ctx-copy').addEventListener('click', () => {
          if (contextMenuTarget) {
            navigator.clipboard.writeText(contextMenuTarget.dataset.path);
            showNotification('success', 'Path Copied', contextMenuTarget.dataset.path);
          }
          hideContextMenu();
        });

        document.getElementById('ctx-copy-rel').addEventListener('click', () => {
          if (contextMenuTarget) {
            navigator.clipboard.writeText(contextMenuTarget.dataset.path);
            showNotification('success', 'Relative Path Copied', contextMenuTarget.dataset.path);
          }
          hideContextMenu();
        });

        // Window controls
        document.getElementById('minimize-btn').addEventListener('click', () => {
          // In a real app, this would minimize the window
          showNotification('info', 'Minimize', 'Window would be minimized');
        });

        document.getElementById('maximize-btn').addEventListener('click', () => {
          // In a real app, this would maximize the window
          showNotification('info', 'Maximize', 'Window would be maximized');
        });

        document.getElementById('close-btn').addEventListener('click', () => {
          // In a real app, this would close the window
          if (confirm('Are you sure you want to close Cursor AI?')) {
            showNotification('info', 'Close', 'Window would be closed');
          }
        });
      }

      function loadFileTree() {
        fetch(`/api/project/${currentProjectId}/files`)
          .then(response => response.json())
          .then(data => {
            if (data.success) {
              renderFileTree(data.files);
            }
          })
          .catch(error => {
            console.error('Error loading file tree:', error);
          });
      }

      function renderFileTree(files, parentElement = document.getElementById('file-tree'), level = 0) {
        parentElement.innerHTML = '';
        
        files.forEach(file => {
          const fileItem = document.createElement('div');
          fileItem.className = 'file-item';
          fileItem.dataset.path = file.path;
          fileItem.style.paddingLeft = `${level * 15}px`;
          
          const icon = document.createElement('i');
          icon.className = 'fas';
          
          const name = document.createElement('span');
          name.textContent = file.name;
          
          fileItem.appendChild(icon);
          fileItem.appendChild(name);
          
          if (file.type === 'directory') {
            fileItem.classList.add('folder');
            
            const childrenContainer = document.createElement('div');
            childrenContainer.className = 'file-children hidden';
            childrenContainer.style.paddingLeft = '15px';
            
            fileItem.addEventListener('click', (e) => {
              e.stopPropagation();
              fileItem.classList.toggle('expanded');
              childrenContainer.classList.toggle('hidden');
              
              if (fileItem.classList.contains('expanded') && !childrenContainer.hasChildNodes()) {
                // Load children on demand
                fetch(`/api/project/${currentProjectId}/files?path=${file.path}`)
                  .then(response => response.json())
                  .then(data => {
                    if (data.success) {
                      renderFileTree(data.files, childrenContainer, level + 1);
                    }
                  })
                  .catch(error => {
                    console.error('Error loading directory contents:', error);
                  });
              }
            });
            
            parentElement.appendChild(fileItem);
            parentElement.appendChild(childrenContainer);
          } else {
            // Determine file type for icon
            const extension = file.name.split('.').pop().toLowerCase();
            if (['py', 'python'].includes(extension)) {
              fileItem.classList.add('python');
            } else if (['js', 'javascript', 'jsx', 'ts', 'tsx'].includes(extension)) {
              fileItem.classList.add('javascript');
            } else if (['html', 'htm'].includes(extension)) {
              fileItem.classList.add('html');
            } else if (['css', 'scss', 'sass', 'less'].includes(extension)) {
              fileItem.classList.add('css');
            } else if (['json'].includes(extension)) {
              fileItem.classList.add('json');
            } else if (['txt', 'md', 'markdown'].includes(extension)) {
              fileItem.classList.add('txt');
            } else {
              fileItem.classList.add('default');
            }
            
            fileItem.addEventListener('click', () => {
              openFile(file.path);
            });
            
            parentElement.appendChild(fileItem);
          }
        });
      }

      function openFile(filePath) {
        // Check if file is already open
        const existingTab = openTabs.find(tab => tab.path === filePath);
        if (existingTab) {
          // Switch to existing tab
          switchToTab(existingTab.id);
          return;
        }
        
        // Load file content
        fetch(`/api/project/${currentProjectId}/file?path=${encodeURIComponent(filePath)}`)
          .then(response => response.json())
          .then(data => {
            if (data.success) {
              // Create new tab
              const tabId = 'tab-' + Date.now();
              const tab = {
                id: tabId,
                name: data.name,
                path: filePath,
                language: data.language,
                content: data.content
              };
              
              openTabs.push(tab);
              createTabElement(tab);
              switchToTab(tabId);
              
              // Update file type in status bar
              document.getElementById('file-type').textContent = data.language || 'Plain Text';
            } else {
              showNotification('error', 'Error', data.message || 'Failed to open file');
            }
          })
          .catch(error => {
            console.error('Error opening file:', error);
            showNotification('error', 'Error', 'Failed to open file');
          });
      }

      function createTabElement(tab) {
        const tabsContainer = document.getElementById('tabs-container');
        
        const tabElement = document.createElement('div');
        tabElement.className = 'tab';
        tabElement.id = tab.id;
        
        const tabName = document.createElement('div');
        tabName.className = 'tab-name';
        tabName.textContent = tab.name;
        
        const tabClose = document.createElement('div');
        tabClose.className = 'tab-close';
        tabClose.innerHTML = '&times;';
        
        tabElement.appendChild(tabName);
        tabElement.appendChild(tabClose);
        
        tabElement.addEventListener('click', (e) => {
          if (!e.target.classList.contains('tab-close')) {
            switchToTab(tab.id);
          }
        });
        
        tabClose.addEventListener('click', (e) => {
          e.stopPropagation();
          closeTab(tab.id);
        });
        
        tabsContainer.appendChild(tabElement);
      }

      function switchToTab(tabId) {
        // Update active tab
        activeTab = tabId;
        
        // Update tab UI
        document.querySelectorAll('.tab').forEach(tab => {
          tab.classList.remove('active');
        });
        document.getElementById(tabId).classList.add('active');
        
        // Update editor content
        const tab = openTabs.find(t => t.id === tabId);
        if (tab) {
          editor.setValue(tab.content);
          monaco.editor.setModelLanguage(editor.getModel(), tab.language || 'plaintext');
        }
      }

      function closeTab(tabId) {
        const tabIndex = openTabs.findIndex(t => t.id === tabId);
        if (tabIndex === -1) return;
        
        // Save current content to tab
        const tab = openTabs[tabIndex];
        tab.content = editor.getValue();
        
        // Remove tab
        openTabs.splice(tabIndex, 1);
        document.getElementById(tabId).remove();
        
        // If this was the active tab, switch to another
        if (activeTab === tabId) {
          if (openTabs.length > 0) {
            switchToTab(openTabs[Math.max(0, tabIndex - 1)].id);
          } else {
            // No tabs left, create a new empty one
            editor.setValue('// No file open\\n');
            activeTab = null;
          }
        }
        
        // Save file content
        saveFile(tab.path, tab.content);
      }

      function saveFile(filePath, content) {
        fetch(`/api/project/${currentProjectId}/file`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: filePath, content })
        })
        .then(response => response.json())
        .then(data => {
          if (!data.success) {
            showNotification('error', 'Error', data.message || 'Failed to save file');
          }
        })
        .catch(error => {
          console.error('Error saving file:', error);
          showNotification('error', 'Error', 'Failed to save file');
        });
      }

      function createNewFile() {
        const name = document.getElementById('new-file-name').value.trim();
        const path = document.getElementById('new-file-path').value.trim();
        
        if (!name) {
          showNotification('error', 'Error', 'Please enter a file name');
          return;
        }
        
        const fullPath = path ? `${path}${name}` : name;
        
        fetch(`/api/project/${currentProjectId}/file`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: fullPath })
        })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            showNotification('success', 'Success', 'File created successfully');
            document.getElementById('new-file-modal').classList.add('hidden');
            document.getElementById('new-file-name').value = '';
            document.getElementById('new-file-path').value = '';
            loadFileTree();
            openFile(fullPath);
          } else {
            showNotification('error', 'Error', data.message || 'Failed to create file');
          }
        })
        .catch(error => {
          console.error('Error creating file:', error);
          showNotification('error', 'Error', 'Failed to create file');
        });
      }

      function createNewFolder() {
        const name = document.getElementById('new-folder-name').value.trim();
        const path = document.getElementById('new-folder-path').value.trim();
        
        if (!name) {
          showNotification('error', 'Error', 'Please enter a folder name');
          return;
        }
        
        const fullPath = path ? `${path}${name}` : name;
        
        fetch(`/api/project/${currentProjectId}/folder`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: fullPath })
        })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            showNotification('success', 'Success', 'Folder created successfully');
            document.getElementById('new-folder-modal').classList.add('hidden');
            document.getElementById('new-folder-name').value = '';
            document.getElementById('new-folder-path').value = '';
            loadFileTree();
          } else {
            showNotification('error', 'Error', data.message || 'Failed to create folder');
          }
        })
        .catch(error => {
          console.error('Error creating folder:', error);
          showNotification('error', 'Error', 'Failed to create folder');
        });
      }

      function renameFile(oldPath, newName) {
        // This would need to be implemented on the backend
        showNotification('info', 'Rename', `File ${oldPath} would be renamed to ${newName}`);
      }

      function deleteFile(path) {
        fetch(`/api/project/${currentProjectId}/delete`, {
          method: 'DELETE',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path })
        })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            showNotification('success', 'Success', 'File deleted successfully');
            loadFileTree();
            
            // Close tab if file was open
            const tabIndex = openTabs.findIndex(t => t.path === path);
            if (tabIndex !== -1) {
              closeTab(openTabs[tabIndex].id);
            }
          } else {
            showNotification('error', 'Error', data.message || 'Failed to delete file');
          }
        })
        .catch(error => {
          console.error('Error deleting file:', error);
          showNotification('error', 'Error', 'Failed to delete file');
        });
      }

      function runCode() {
        const language = document.getElementById('debug-language').value;
        const filePath = activeTab ? openTabs.find(t => t.id === activeTab).path : null;
        const code = activeTab ? openTabs.find(t => t.id === activeTab).content : editor.getValue();
        
        if (!code.trim()) {
          showNotification('error', 'Error', 'No code to run');
          return;
        }
        
        // Show terminal
        document.querySelectorAll('.panel-tab').forEach(t => t.classList.remove('active'));
        document.querySelector('[data-panel="terminal"]').classList.add('active');
        document.querySelectorAll('.terminal, .ai-chat').forEach(p => p.classList.add('hidden'));
        document.getElementById('terminal-panel').classList.remove('hidden');
        
        // Add running message to terminal
        const terminalContent = document.getElementById('terminal-content');
        terminalContent.innerHTML += `<div>Running ${language} code...</div>`;
        
        fetch('/api/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            project_id: currentProjectId,
            language,
            code,
            file_path: filePath
          })
        })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            terminalContent.innerHTML += `<div>Exit code: ${data.exit_code}</div>`;
            if (data.stdout) {
              terminalContent.innerHTML += `<div>STDOUT:</div><div>${data.stdout}</div>`;
            }
            if (data.stderr) {
              terminalContent.innerHTML += `<div>STDERR:</div><div>${data.stderr}</div>`;
            }
          } else {
            terminalContent.innerHTML += `<div>Error: ${data.error || 'Failed to run code'}</div>`;
          }
          terminalContent.innerHTML += `<div>$ </div>`;
          terminalContent.scrollTop = terminalContent.scrollHeight;
        })
        .catch(error => {
          console.error('Error running code:', error);
          terminalContent.innerHTML += `<div>Error: Failed to run code</div><div>$ </div>`;
          terminalContent.scrollTop = terminalContent.scrollHeight;
        });
      }

      function installPackage() {
        const packageManager = document.getElementById('package-manager').value;
        const packages = document.getElementById('package-name').value.trim();
        
        if (!packages) {
          showNotification('error', 'Error', 'Please enter package name(s)');
          return;
        }
        
        // Show output panel
        document.querySelectorAll('.panel-tab').forEach(t => t.classList.remove('active'));
        document.querySelector('[data-panel="output"]').classList.add('active');
        document.querySelectorAll('.terminal, .ai-chat').forEach(p => p.classList.add('hidden'));
        document.getElementById('output-panel').classList.remove('hidden');
        
        // Add installing message to output
        const outputContent = document.getElementById('output-content');
        outputContent.innerHTML = `<div>Installing packages with ${packageManager}...</div>`;
        
        fetch('/api/install', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            project_id: currentProjectId,
            package_manager: packageManager,
            packages
          })
        })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            outputContent.innerHTML += `<div>Packages installed successfully!</div>`;
            if (data.stdout) {
              outputContent.innerHTML += `<div>${data.stdout}</div>`;
            }
          } else {
            outputContent.innerHTML += `<div>Error: ${data.error || 'Failed to install packages'}</div>`;
            if (data.stderr) {
              outputContent.innerHTML += `<div>${data.stderr}</div>`;
            }
          }
        })
        .catch(error => {
          console.error('Error installing packages:', error);
          outputContent.innerHTML += `<div>Error: Failed to install packages</div>`;
        });
      }

      function sendChatMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message) return;
        
        // Add user message to chat
        addChatMessage('user', message);
        input.value = '';
        
        // Add loading message
        const loadingId = 'loading-' + Date.now();
        addChatMessage('assistant', 'Thinking...', loadingId);
        
        fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message })
        })
        .then(response => response.json())
        .then(data => {
          // Remove loading message
          document.getElementById(loadingId)?.remove();
          
          if (data.error) {
            addChatMessage('assistant', `Error: ${data.error}`);
          } else {
            addChatMessage('assistant', data.reply || data.text || '');
          }
        })
        .catch(error => {
          console.error('Error sending chat message:', error);
          document.getElementById(loadingId)?.remove();
          addChatMessage('assistant', 'Error: Failed to get response from AI');
        });
      }

      function addChatMessage(role, content, id = null) {
        const messagesContainer = document.getElementById('chat-messages');
        
        const messageElement = document.createElement('div');
        messageElement.className = `chat-message ${role}`;
        if (id) messageElement.id = id;
        
        const contentElement = document.createElement('div');
        contentElement.className = 'chat-message-content';
        contentElement.textContent = content;
        
        messageElement.appendChild(contentElement);
        messagesContainer.appendChild(messageElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
      }

      function showContextMenu(x, y) {
        const contextMenu = document.getElementById('context-menu');
        contextMenu.style.left = `${x}px`;
        contextMenu.style.top = `${y}px`;
        contextMenu.style.display = 'block';
      }

      function hideContextMenu() {
        document.getElementById('context-menu').style.display = 'none';
      }

      function showNotification(type, title, message) {
        const notification = document.getElementById('notification');
        const titleElement = document.getElementById('notification-title');
        const messageElement = document.getElementById('notification-message');
        
        // Set notification type
        notification.className = 'notification';
        notification.classList.add(type);
        
        // Set content
        titleElement.textContent = title;
        messageElement.textContent = message;
        
        // Show notification
        notification.classList.add('show');
        
        // Hide after 3 seconds
        setTimeout(() => {
          notification.classList.remove('show');
        }, 3000);
      }
    });
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

# -----------------------
# API Routes
# -----------------------

@app.route("/api/project/init", methods=["POST"])
def init_project():
    data = request.get_json(force=True)
    project_name = data.get("project_name", "Untitled Project")
    
    # Generate unique project ID
    project_id = str(uuid.uuid4())
    
    # Create project directory
    project_dir = create_project(project_id, project_name)
    
    return jsonify({
        "success": True,
        "project_id": project_id,
        "project_name": project_name
    })

@app.route("/api/project/<project_id>/files")
def get_files(project_id):
    path = request.args.get("path", "")
    files = get_project_files(project_id, path)
    return jsonify({
        "success": True,
        "files": files
    })

@app.route("/api/project/<project_id>/file", methods=["GET", "POST"])
def handle_file(project_id):
    if request.method == "GET":
        file_path = request.args.get("path", "")
        if not file_path:
            return jsonify({"success": False, "message": "No file path provided"})
        
        content = get_file_content(project_id, file_path)
        if content is None:
            return jsonify({"success": False, "message": "File not found"})
        
        # Determine language from file extension
        extension = file_path.split(".")[-1].lower()
        language_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "jsx": "javascript",
            "tsx": "typescript",
            "html": "html",
            "css": "css",
            "scss": "scss",
            "sass": "sass",
            "less": "less",
            "json": "json",
            "md": "markdown",
            "sh": "shell",
            "bash": "shell"
        }
        language = language_map.get(extension, "plaintext")
        
        return jsonify({
            "success": True,
            "name": os.path.basename(file_path),
            "content": content,
            "language": language
        })
    
    elif request.method == "POST":
        data = request.get_json(force=True)
        file_path = data.get("path", "")
        content = data.get("content", "")
        
        if not file_path:
            return jsonify({"success": False, "message": "No file path provided"})
        
        success = save_file_content(project_id, file_path, content)
        return jsonify({
            "success": success,
            "message": "File saved successfully" if success else "Failed to save file"
        })

@app.route("/api/project/<project_id>/file", methods=["PUT"])
def create_file(project_id):
    data = request.get_json(force=True)
    file_path = data.get("path", "")
    content = data.get("content", "")
    
    if not file_path:
        return jsonify({"success": False, "message": "No file path provided"})
    
    success, message = create_file(project_id, file_path, content)
    return jsonify({
        "success": success,
        "message": message
    })

@app.route("/api/project/<project_id>/folder", methods=["PUT"])
def create_folder(project_id):
    data = request.get_json(force=True)
    dir_path = data.get("path", "")
    
    if not dir_path:
        return jsonify({"success": False, "message": "No directory path provided"})
    
    success, message = create_directory(project_id, dir_path)
    return jsonify({
        "success": success,
        "message": message
    })

@app.route("/api/project/<project_id>/delete", methods=["DELETE"])
def delete_file_or_dir(project_id):
    data = request.get_json(force=True)
    path = data.get("path", "")
    
    if not path:
        return jsonify({"success": False, "message": "No path provided"})
    
    success, message = delete_file_or_dir(project_id, path)
    return jsonify({
        "success": success,
        "message": message
    })

@app.route("/api/run", methods=["POST"])
def run_endpoint():
    data = request.get_json(force=True)
    project_id = data.get("project_id", "default")
    language = data.get("language", "python")
    code = data.get("code", "")
    file_path = data.get("file_path", "")
    
    if not code:
        return jsonify({"success": False, "error": "No code provided"}), 400
    
    res = run_code(project_id, language, code, file_path, timeout_seconds=15)
    return jsonify(res)

@app.route("/api/install", methods=["POST"])
def install_endpoint():
    data = request.get_json(force=True)
    project_id = data.get("project_id", "default")
    package_manager = data.get("package_manager", "pip")
    packages = data.get("packages", "")
    
    if not packages:
        return jsonify({"success": False, "error": "No packages provided"}), 400
    
    res = install_dependencies(project_id, package_manager, packages)
    return jsonify(res)

@app.route("/api/chat", methods=["POST"])
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

@app.route("/api/autofix", methods=["POST"])
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
    print(f"Starting Cursor AI on 0.0.0.0:{port} debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)
