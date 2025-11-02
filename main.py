#!/usr/bin/env python3
"""
main.py - Flask chat app (emoji + large input + restartable chat)
Requires: NotoColorEmoji.ttf in same directory
"""

from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context, send_from_directory
import requests
import json
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit for long user inputs

# --- INFERENCE CONFIG ---
INFERENCE_URL = os.getenv("INFERENCE_URL", "https://api.openai.com/v1/chat/completions")
INFERENCE_KEY = os.getenv("INFERENCE_KEY", "")
INFERENCE_MODEL_ID = os.getenv("INFERENCE_MODEL_ID", "gpt-3.5-turbo")

# --- HTML TEMPLATE ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chat</title>
<style>
  @font-face {
    font-family: 'NotoColorEmoji';
    src: url('/font/NotoColorEmoji.ttf') format('truetype');
  }
  body {
    background: #111;
    color: #fff;
    font-family: system-ui, 'NotoColorEmoji', sans-serif;
    margin: 0;
    padding: 0;
  }
  #chat {
    max-width: 700px;
    margin: auto;
    padding: 1rem;
    font-size: 1rem;
    line-height: 1.5;
    white-space: pre-wrap;
    overflow-wrap: break-word;
  }
  #input {
    width: 100%;
    box-sizing: border-box;
    padding: 10px;
    font-size: 1rem;
    font-family: system-ui, 'NotoColorEmoji', sans-serif;
    border: none;
    outline: none;
    border-top: 1px solid #333;
    background: #000;
    color: #fff;
  }
  .msg { margin: 0.5em 0; }
  .user { color: #0f0; }
  .bot { color: #0cf; }
  button {
    background: #0cf;
    color: #000;
    border: none;
    padding: 8px 12px;
    margin-top: 8px;
    cursor: pointer;
    border-radius: 4px;
    font-weight: bold;
  }
</style>
</head>
<body>
  <div id="chat"></div>
  <textarea id="input" placeholder="Type your message..."></textarea>
  <button onclick="sendMessage()">Send</button>
  <button onclick="newChat()">New Chat</button>

  <script>
    let chat = [];
    const chatDiv = document.getElementById('chat');
    const input = document.getElementById('input');

    function renderChat() {
      chatDiv.innerHTML = chat.map(m =>
        `<div class='msg ${m.role}'>${m.role === 'user' ? 'You' : 'AI'}: ${m.content}</div>`
      ).join('');
      window.scrollTo(0, document.body.scrollHeight);
    }

    async function sendMessage() {
      const content = input.value.trim();
      if (!content) return;
      chat.push({ role: 'user', content });
      renderChat();
      input.value = '';

      const res = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({messages: chat})
      });

      const reader = res.body.getReader();
      let botMsg = { role: 'bot', content: '' };
      chat.push(botMsg);
      renderChat();

      const decoder = new TextDecoder('utf-8');
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        botMsg.content += decoder.decode(value, {stream: true});
        renderChat();
      }
    }

    function newChat() {
      chat = [];
      renderChat();
    }
  </script>
</body>
</html>
"""

# --- ROUTES ---
@app.route("/")
def home():
    return Response(HTML_PAGE, content_type="text/html; charset=utf-8")

@app.route("/font/<path:filename>")
def serve_font(filename):
    return send_from_directory(".", filename)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        messages = data.get("messages", [])
        payload = {
            "model": INFERENCE_MODEL_ID,
            "messages": messages,
            "stream": True
        }

        headers = {
            "Authorization": f"Bearer {INFERENCE_KEY}",
            "Content-Type": "application/json"
        }

        r = requests.post(INFERENCE_URL, headers=headers, json=payload, stream=True, timeout=600)

        def generate():
            for chunk in r.iter_lines(decode_unicode=True):
                if chunk:
                    try:
                        if chunk.startswith("data: "):
                            j = json.loads(chunk[6:])
                            delta = j["choices"][0]["delta"].get("content", "")
                            yield delta
                    except Exception:
                        continue
        return Response(stream_with_context(generate()), content_type="text/plain; charset=utf-8")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
