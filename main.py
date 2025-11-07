import requests
import sseclient
import json
from flask import Flask, render_template_string, request, Response, jsonify
import time
import uuid
import logging
from logging.handlers import RotatingFileHandler
import os

# ================== LOGGING CONFIG ==================
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('chat_app.log', maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


INFERENCE_KEY = os.getenv("INFERENCE_KEY")
INFERENCE_BASE_URL = os.getenv("INFERENCE_URL", "https://us.inference.heroku.com")
MODEL = os.getenv("INFERENCE_MODEL_ID", "claude-4-5-sonnet")

# Ensure correct endpoint formatting
if INFERENCE_BASE_URL.endswith("/"):
    INFERENCE_URL = f"{INFERENCE_BASE_URL}v1/chat/completions"
else:
    INFERENCE_URL = f"{INFERENCE_BASE_URL}/v1/chat/completions"

# Validate env vars
if not INFERENCE_KEY:
    raise ValueError("INFERENCE_KEY environment variable not set")

headers = {
    "Authorization": f"Bearer {INFERENCE_KEY}",
    "Content-Type": "application/json",
    "Accept": "text/event-stream"
}

# Store conversations in memory (in production, use a database)
conversations = {}
# Store session metadata
session_metadata = {}


# ================== TOOLS CONFIG ==================
# Define available tools/functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetches the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get weather for."
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Performs mathematical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2+2', '10*5')."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Searches the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Tool execution functions
def execute_tool(tool_name, parameters):
    """Execute a tool with the given parameters."""
    if tool_name == "get_weather":
        # Validate required parameters
        if "city" not in parameters or not parameters["city"]:
            return {
                "error": "Missing required parameter: city",
                "example": {
                    "name": "get_weather",
                    "parameters": {"city": "New York"}
                }
            }
        
        # Mock weather function - in a real app, you'd call a weather API
        city = parameters.get("city", "Unknown")
        return f"The weather in {city} is currently 72°F with partly cloudy skies."
    
    elif tool_name == "calculate":
        # Validate required parameters
        if "expression" not in parameters or not parameters["expression"]:
            return {
                "error": "Missing required parameter: expression",
                "example": {
                    "name": "calculate",
                    "parameters": {"expression": "2+2"}
                }
            }
        
        try:
            expression = parameters.get("expression", "")
            # In a real app, you'd use a safer evaluation method
            result = eval(expression)
            return f"The result of {expression} is {result}."
        except Exception as e:
            return f"Error calculating {expression}: {str(e)}"
    
    elif tool_name == "search_web":
        # Validate required parameters
        if "query" not in parameters or not parameters["query"]:
            return {
                "error": "Missing required parameter: query",
                "example": {
                    "name": "search_web",
                    "parameters": {"query": "latest news"}
                }
            }
        
        # Mock search function - in a real app, you'd call a search API
        query = parameters.get("query", "")
        return f"Search results for '{query}': This is a mock search result. In a real implementation, this would connect to a search API."
    
    else:
        return f"Unknown tool: {tool_name}"

# ================== WEB APP ==================
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claude 4.5 Premium Chat</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.2/marked.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
    <style>
        :root {
            --primary-color: #6366f1;
            --primary-dark: #4f46e5;
            --secondary-color: #22d3ee;
            --bg-color: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-color: #e2e8f0;
            --text-secondary: #94a3b8;
            --border-color: #334155;
            --shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            --code-bg: #1e293b;
            --code-border: #475569;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, var(--bg-color) 0%, #1a1f3a 100%);
            color: var(--text-color);
            height: 100vh;
            overflow: hidden;
        }

        .chat-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            max-width: 1200px;
            margin: 0 auto;
            box-shadow: var(--shadow-lg);
            background: var(--bg-secondary);
            border-left: 1px solid var(--border-color);
            border-right: 1px solid var(--border-color);
        }

        .chat-header {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
            padding: 1rem 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: var(--shadow);
            z-index: 10;
        }

        .header-title {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .header-title h1 {
            font-size: 1.5rem;
            font-weight: 600;
        }

        .header-title .logo {
            width: 40px;
            height: 40px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .header-actions {
            display: flex;
            gap: 0.75rem;
        }

        .header-actions button {
            background: rgba(255, 255, 255, 0.1);
            border: none;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .header-actions button:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: scale(1.05);
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            background: var(--bg-color);
        }

        .message {
            display: flex;
            gap: 0.75rem;
            max-width: 80%;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            align-self: flex-end;
            flex-direction: row-reverse;
        }

        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .user .message-avatar {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        }

        .bot .message-avatar {
            background: linear-gradient(135deg, var(--secondary-color) 0%, #0891b2 100%);
        }

        .message-content {
            background: var(--bg-tertiary);
            padding: 0.75rem 1rem;
            border-radius: 1rem;
            position: relative;
            box-shadow: var(--shadow);
            width: 100%;
            overflow: hidden;
        }

        .user .message-content {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
            border-bottom-right-radius: 0.25rem;
        }

        .bot .message-content {
            background: var(--bg-tertiary);
            border-bottom-left-radius: 0.25rem;
        }

        .message-text {
            word-wrap: break-word;
            line-height: 1.5;
        }

        .message-text p {
            margin-bottom: 0.75rem;
        }

        .message-text p:last-child {
            margin-bottom: 0;
        }

        .message-text ul, .message-text ol {
            margin-left: 1.5rem;
            margin-bottom: 0.75rem;
        }

        .message-text li {
            margin-bottom: 0.25rem;
        }

        .message-text h1, .message-text h2, .message-text h3, 
        .message-text h4, .message-text h5, .message-text h6 {
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }

        .message-text h1:first-child, .message-text h2:first-child, 
        .message-text h3:first-child, .message-text h4:first-child, 
        .message-text h5:first-child, .message-text h6:first-child {
            margin-top: 0;
        }

        .message-text blockquote {
            border-left: 4px solid var(--primary-color);
            padding-left: 1rem;
            margin: 0.75rem 0;
            color: var(--text-secondary);
            font-style: italic;
        }

        .message-text pre {
            background: var(--code-bg);
            border-radius: 0.5rem;
            padding: 1rem;
            overflow-x: auto;
            margin: 0.75rem 0;
            border: 1px solid var(--code-border);
            max-height: 500px;
            overflow-y: auto;
            position: relative;
        }

        .message-text code {
            background: rgba(99, 102, 241, 0.1);
            color: #a5b4fc;
            padding: 0.125rem 0.25rem;
            border-radius: 0.25rem;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
        }

        .message-text pre code {
            background: transparent;
            color: inherit;
            padding: 0;
            border-radius: 0;
        }

        .message-text table {
            border-collapse: collapse;
            width: 100%;
            margin: 0.75rem 0;
        }

        .message-text th, .message-text td {
            border: 1px solid var(--border-color);
            padding: 0.5rem;
            text-align: left;
        }

        .message-text th {
            background: var(--bg-secondary);
        }

        .message-time {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
            text-align: right;
        }

        .typing-indicator {
            display: flex;
            gap: 0.25rem;
            padding: 0.75rem 1rem;
        }

        .typing-indicator span {
            width: 8px;
            height: 8px;
            background-color: var(--text-secondary);
            border-radius: 50%;
            display: inline-block;
            animation: typing 1.4s infinite ease-in-out;
        }

        .typing-indicator span:nth-child(1) {
            animation-delay: -0.32s;
        }

        .typing-indicator span:nth-child(2) {
            animation-delay: -0.16s;
        }

        @keyframes typing {
            0%, 80%, 100% {
                transform: scale(0.8);
                opacity: 0.5;
            }
            40% {
                transform: scale(1);
                opacity: 1;
            }
        }

        .tool-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1rem;
            color: var(--secondary-color);
        }

        .tool-indicator i {
            animation: pulse 1.5s infinite ease-in-out;
        }

        @keyframes pulse {
            0%, 100% {
                opacity: 0.5;
            }
            50% {
                opacity: 1;
            }
        }

        .copy-button {
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background: rgba(255, 255, 255, 0.1);
            border: none;
            color: var(--text-secondary);
            width: 30px;
            height: 30px;
            border-radius: 0.25rem;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
            opacity: 0;
        }

        .message-text pre:hover .copy-button {
            opacity: 1;
        }

        .copy-button:hover {
            background: rgba(255, 255, 255, 0.2);
            color: var(--text-color);
        }

        .chat-input-container {
            padding: 1rem 1.5rem;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border-color);
        }

        .chat-input-form {
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }

        .chat-input {
            flex: 1;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 2rem;
            padding: 0.75rem 1.25rem;
            color: var(--text-color);
            font-size: 1rem;
            outline: none;
            transition: all 0.2s ease;
        }

        .chat-input:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }

        .chat-input::placeholder {
            color: var(--text-secondary);
        }

        .chat-actions {
            display: flex;
            gap: 0.5rem;
        }

        .chat-action-btn {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .chat-action-btn:hover {
            background: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
            transform: scale(1.05);
        }

        .send-btn {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
            border: none;
            color: white;
        }

        .send-btn:hover {
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-color) 100%);
        }

        .welcome-message {
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
        }

        .welcome-message i {
            font-size: 3rem;
            margin-bottom: 1rem;
            color: var(--primary-color);
        }

        .welcome-message h2 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
            color: var(--text-color);
        }

        .error-message {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #f87171;
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }

        .tool-call {
            background: rgba(34, 211, 238, 0.1);
            border: 1px solid rgba(34, 211, 238, 0.3);
            color: #22d3ee;
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            font-family: monospace;
            font-size: 0.9rem;
        }

        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(15, 23, 42, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s ease, visibility 0.3s ease;
        }

        .loading-overlay.active {
            opacity: 1;
            visibility: visible;
        }

        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(99, 102, 241, 0.2);
            border-top: 4px solid var(--primary-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .session-modal {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(15, 23, 42, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s ease, visibility 0.3s ease;
        }

        .session-modal.active {
            opacity: 1;
            visibility: visible;
        }

        .session-modal-content {
            background: var(--bg-secondary);
            border-radius: 0.5rem;
            width: 90%;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: var(--shadow-lg);
            border: 1px solid var(--border-color);
        }

        .session-modal-header {
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .session-modal-title {
            font-size: 1.25rem;
            font-weight: 600;
        }

        .session-modal-close {
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 1.5rem;
            cursor: pointer;
            transition: color 0.2s ease;
        }

        .session-modal-close:hover {
            color: var(--text-color);
        }

        .session-modal-body {
            padding: 1.5rem;
        }

        .session-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .session-item {
            background: var(--bg-tertiary);
            border-radius: 0.5rem;
            padding: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: background 0.2s ease;
        }

        .session-item:hover {
            background: rgba(99, 102, 241, 0.1);
        }

        .session-item.active {
            background: rgba(99, 102, 241, 0.2);
            border: 1px solid var(--primary-color);
        }

        .session-info {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .session-name {
            font-weight: 500;
        }

        .session-date {
            font-size: 0.875rem;
            color: var(--text-secondary);
        }

        .session-actions {
            display: flex;
            gap: 0.5rem;
        }

        .session-action-btn {
            background: rgba(255, 255, 255, 0.1);
            border: none;
            color: var(--text-secondary);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .session-action-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            color: var(--text-color);
        }

        .session-action-btn.delete:hover {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
        }

        .new-session-btn {
            background: var(--primary-color);
            border: none;
            color: white;
            border-radius: 0.5rem;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s ease;
            margin-bottom: 1rem;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .new-session-btn:hover {
            background: var(--primary-dark);
        }

        /* Scrollbar styling */
        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }

        .chat-messages::-webkit-scrollbar-track {
            background: var(--bg-secondary);
        }

        .chat-messages::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 3px;
        }

        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: var(--text-secondary);
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .chat-container {
                height: 100vh;
                max-width: 100%;
                border-left: none;
                border-right: none;
            }

            .chat-header {
                padding: 0.75rem 1rem;
            }

            .header-title h1 {
                font-size: 1.25rem;
            }

            .chat-messages {
                padding: 1rem;
            }

            .message {
                max-width: 90%;
            }

            .chat-input-container {
                padding: 0.75rem 1rem;
            }

            .chat-action-btn {
                width: 40px;
                height: 40px;
            }
        }

        @media (max-width: 480px) {
            .header-title h1 {
                display: none;
            }

            .message {
                max-width: 95%;
            }

            .message-content {
                padding: 0.5rem 0.75rem;
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div class="header-title">
                <div class="logo">
                    <i class="fas fa-robot"></i>
                </div>
                <h1>Claude 4.5 Assistant</h1>
            </div>
            <div class="header-actions">
                <button id="sessions-btn" title="Sessions">
                    <i class="fas fa-history"></i>
                </button>
                <button id="clear-chat" title="Clear Chat">
                    <i class="fas fa-trash"></i>
                </button>
                <button id="settings-btn" title="Settings">
                    <i class="fas fa-cog"></i>
                </button>
            </div>
        </div>

        <div class="chat-messages" id="chat-messages">
            <div class="welcome-message">
                <i class="fas fa-comments"></i>
                <h2>Welcome to Claude 4.5 Assistant</h2>
                <p>I'm here to help you with any questions or tasks you have. Just type your message below and press send.</p>
            </div>
        </div>

        <div class="chat-input-container">
            <form class="chat-input-form" id="chat-form">
                <input type="text" class="chat-input" id="user-input" placeholder="Type your message..." autocomplete="off" maxlength="5000">
                <div class="chat-actions">
                    <button type="button" class="chat-action-btn" id="attach-btn" title="Attach File">
                        <i class="fas fa-paperclip"></i>
                    </button>
                    <button type="submit" class="chat-action-btn send-btn" id="send-btn" title="Send Message">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
            </form>
        </div>
    </div>

    <div class="loading-overlay" id="loading-overlay">
        <div class="loading-spinner"></div>
    </div>

    <div class="session-modal" id="session-modal">
        <div class="session-modal-content">
            <div class="session-modal-header">
                <h2 class="session-modal-title">Chat Sessions</h2>
                <button class="session-modal-close" id="session-modal-close">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="session-modal-body">
                <button class="new-session-btn" id="new-session-btn">
                    <i class="fas fa-plus"></i>
                    New Chat Session
                </button>
                <div class="session-list" id="session-list">
                    <!-- Sessions will be populated here -->
                </div>
            </div>
        </div>
    </div>

    <script>
        // Generate a unique session ID for this user
        let sessionId = localStorage.getItem('chatSessionId') || generateSessionId();
        localStorage.setItem('chatSessionId', sessionId);

        function generateSessionId() {
            return 'session_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
        }

        // Configure marked for better code handling
        marked.setOptions({
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (err) {}
                }
                return hljs.highlightAuto(code).value;
            },
            langPrefix: 'hljs language-',
            breaks: true,
            gfm: true
        });

        // DOM elements
        const chatMessages = document.getElementById('chat-messages');
        const chatForm = document.getElementById('chat-form');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const clearChatBtn = document.getElementById('clear-chat');
        const attachBtn = document.getElementById('attach-btn');
        const settingsBtn = document.getElementById('settings-btn');
        const sessionsBtn = document.getElementById('sessions-btn');
        const loadingOverlay = document.getElementById('loading-overlay');
        const sessionModal = document.getElementById('session-modal');
        const sessionModalClose = document.getElementById('session-modal-close');
        const newSessionBtn = document.getElementById('new-session-btn');
        const sessionList = document.getElementById('session-list');

        // Remove welcome message when first message is sent
        let welcomeMessageRemoved = false;
        let isProcessing = false;
        let currentMessageId = null;
        let updateTimeout = null;
        let lastUpdateTime = 0;
        const UPDATE_THROTTLE_MS = 100; // Throttle UI updates to every 100ms

        // Event listeners
        chatForm.addEventListener('submit', sendMessage);
        clearChatBtn.addEventListener('click', clearChat);
        attachBtn.addEventListener('click', () => {
            showNotification('File attachment feature coming soon!');
        });
        settingsBtn.addEventListener('click', () => {
            showNotification('Settings panel coming soon!');
        });
        sessionsBtn.addEventListener('click', showSessionModal);
        sessionModalClose.addEventListener('click', hideSessionModal);
        newSessionBtn.addEventListener('click', createNewSession);

        // Close modal when clicking outside
        sessionModal.addEventListener('click', (e) => {
            if (e.target === sessionModal) {
                hideSessionModal();
            }
        });

        // Send message function
        async function sendMessage(e) {
            e.preventDefault();
            
            if (isProcessing) return;
            
            const message = userInput.value.trim();
            if (!message) return;

            // Log message length for debugging
            console.log(`Sending message with length: ${message.length} characters`);

            // Remove welcome message if it's still there
            if (!welcomeMessageRemoved) {
                const welcomeMsg = document.querySelector('.welcome-message');
                if (welcomeMsg) welcomeMsg.remove();
                welcomeMessageRemoved = true;
            }

            // Add user message to chat
            addMessage(message, 'user');
            
            // Clear input
            userInput.value = '';
            
            // Set processing state
            isProcessing = true;
            sendBtn.disabled = true;
            sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            
            // Show typing indicator
            const typingId = showTypingIndicator();
            
            try {
                // Show loading overlay for very long messages
                if (message.length > 10000) {
                    loadingOverlay.classList.add('active');
                }
                
                // Send message to server
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Session-ID': sessionId
                    },
                    body: JSON.stringify({ message })
                });
                
                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }
                
                // Remove typing indicator
                removeTypingIndicator(typingId);
                
                // Create bot message element
                currentMessageId = addMessage('', 'bot', true);
                
                // Process streaming response
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let botMessage = '';
                let buffer = '';
                let chunksProcessed = 0;
                let isUsingTool = false;
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value, { stream: true });
                    buffer += chunk;
                    chunksProcessed++;
                    
                    // Log progress for long messages
                    if (chunksProcessed % 10 === 0) {
                        console.log(`Processed ${chunksProcessed} chunks, current message length: ${botMessage.length}`);
                    }
                    
                    // Process SSE events in the buffer
                    const lines = buffer.split('\\n');
                    buffer = lines.pop() || ''; // Keep the last incomplete line in the buffer
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.substring(6));
                                
                                // Check if AI is using a tool
                                if (data.choices && data.choices[0] && data.choices[0].delta && data.choices[0].delta.tool_calls) {
                                    if (!isUsingTool) {
                                        isUsingTool = true;
                                        removeTypingIndicator(typingId);
                                        const toolId = showToolIndicator();
                                        typingId = toolId; // Reuse the variable to track the tool indicator
                                    }
                                } else if (data.choices && data.choices[0] && data.choices[0].delta && data.choices[0].delta.content) {
                                    if (isUsingTool) {
                                        isUsingTool = false;
                                        removeTypingIndicator(typingId);
                                        const newTypingId = showTypingIndicator();
                                        typingId = newTypingId; // Reuse the variable to track the typing indicator
                                    }
                                    
                                    const delta = data.choices[0].delta.content;
                                    if (delta) {
                                        botMessage += delta;
                                        
                                        // Use throttled updates for better performance
                                        const now = Date.now();
                                        if (now - lastUpdateTime > UPDATE_THROTTLE_MS) {
                                            if (updateTimeout) clearTimeout(updateTimeout);
                                            updateTimeout = setTimeout(() => {
                                                updateMessage(currentMessageId, botMessage);
                                                lastUpdateTime = Date.now();
                                            }, 0);
                                        }
                                    }
                                }
                            } catch (e) {
                                console.error('Error parsing SSE data:', e);
                            }
                        }
                    }
                }
                
                // Final update
                if (updateTimeout) clearTimeout(updateTimeout);
                updateMessage(currentMessageId, botMessage);
                console.log(`Final message length: ${botMessage.length} characters`);
                
            } catch (error) {
                console.error('Error:', error);
                removeTypingIndicator(typingId);
                addMessage(`Sorry, I encountered an error while processing your request: ${error.message}. Please try again.`, 'bot', false, true);
            } finally {
                // Reset processing state
                isProcessing = false;
                sendBtn.disabled = false;
                sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
                loadingOverlay.classList.remove('active');
                currentMessageId = null;
            }
        }

        // Add message to chat
        function addMessage(text, sender, isStreaming = false, isError = false) {
            const messageId = 'msg_' + Date.now();
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            messageDiv.id = messageId;
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.innerHTML = sender === 'user' 
                ? '<i class="fas fa-user"></i>' 
                : '<i class="fas fa-robot"></i>';
            
            const content = document.createElement('div');
            content.className = 'message-content';
            
            if (isError) {
                content.classList.add('error-message');
            }
            
            const messageText = document.createElement('div');
            messageText.className = 'message-text';
            
            // Process markdown for bot messages
            if (sender === 'bot' && text) {
                messageText.innerHTML = marked.parse(text);
                // Highlight code blocks
                messageText.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                    
                    // Add copy button to code blocks
                    const pre = block.parentElement;
                    const copyButton = document.createElement('button');
                    copyButton.className = 'copy-button';
                    copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                    copyButton.title = 'Copy code';
                    copyButton.addEventListener('click', () => {
                        navigator.clipboard.writeText(block.textContent).then(() => {
                            copyButton.innerHTML = '<i class="fas fa-check"></i>';
                            setTimeout(() => {
                                copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                            }, 2000);
                        }).catch(err => {
                            console.error('Failed to copy: ', err);
                        });
                    });
                    pre.appendChild(copyButton);
                });
            } else {
                messageText.textContent = text;
            }
            
            const messageTime = document.createElement('div');
            messageTime.className = 'message-time';
            messageTime.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            
            content.appendChild(messageText);
            content.appendChild(messageTime);
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(content);
            
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            return messageId;
        }

        // Update message content with performance optimization
        function updateMessage(messageId, text) {
            if (!messageId) return;
            
            const messageElement = document.getElementById(messageId);
            if (messageElement) {
                const messageText = messageElement.querySelector('.message-text');
                if (messageText) {
                    // Use requestAnimationFrame for smoother updates
                    requestAnimationFrame(() => {
                        // Process markdown for bot messages
                        messageText.innerHTML = marked.parse(text);
                        // Highlight code blocks
                        messageText.querySelectorAll('pre code').forEach((block) => {
                            hljs.highlightElement(block);
                            
                            // Add copy button to code blocks
                            const pre = block.parentElement;
                            if (!pre.querySelector('.copy-button')) {
                                const copyButton = document.createElement('button');
                                copyButton.className = 'copy-button';
                                copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                                copyButton.title = 'Copy code';
                                copyButton.addEventListener('click', () => {
                                    navigator.clipboard.writeText(block.textContent).then(() => {
                                        copyButton.innerHTML = '<i class="fas fa-check"></i>';
                                        setTimeout(() => {
                                            copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                                        }, 2000);
                                    }).catch(err => {
                                        console.error('Failed to copy: ', err);
                                    });
                                });
                                pre.appendChild(copyButton);
                            }
                        });
                        
                        // Auto-scroll to bottom
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    });
                }
            }
        }

        // Show typing indicator
        function showTypingIndicator() {
            const typingId = 'typing_' + Date.now();
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message bot';
            typingDiv.id = typingId;
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.innerHTML = '<i class="fas fa-robot"></i>';
            
            const content = document.createElement('div');
            content.className = 'message-content';
            
            const typingIndicator = document.createElement('div');
            typingIndicator.className = 'typing-indicator';
            typingIndicator.innerHTML = '<span></span><span></span><span></span>';
            
            content.appendChild(typingIndicator);
            typingDiv.appendChild(avatar);
            typingDiv.appendChild(content);
            
            chatMessages.appendChild(typingDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            return typingId;
        }

        // Show tool indicator
        function showToolIndicator() {
            const toolId = 'tool_' + Date.now();
            const toolDiv = document.createElement('div');
            toolDiv.className = 'message bot';
            toolDiv.id = toolId;
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.innerHTML = '<i class="fas fa-robot"></i>';
            
            const content = document.createElement('div');
            content.className = 'message-content';
            
            const toolIndicator = document.createElement('div');
            toolIndicator.className = 'tool-indicator';
            toolIndicator.innerHTML = '<i class="fas fa-cogs"></i> Using a tool...';
            
            content.appendChild(toolIndicator);
            toolDiv.appendChild(avatar);
            toolDiv.appendChild(content);
            
            chatMessages.appendChild(toolDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            return toolId;
        }

        // Remove typing indicator
        function removeTypingIndicator(typingId) {
            const typingElement = document.getElementById(typingId);
            if (typingElement) {
                typingElement.remove();
            }
        }

        // Clear chat
        function clearChat() {
            if (confirm('Are you sure you want to clear the chat history?')) {
                chatMessages.innerHTML = `
                    <div class="welcome-message">
                        <i class="fas fa-comments"></i>
                        <h2>Welcome to Claude 4.5 Assistant</h2>
                        <p>I'm here to help you with any questions or tasks you have. Just type your message below and press send.</p>
                    </div>
                `;
                welcomeMessageRemoved = false;
                
                // Notify server to clear conversation
                fetch('/clear-chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Session-ID': sessionId
                    }
                });
            }
        }

        // Session management functions
        function showSessionModal() {
            loadSessions();
            sessionModal.classList.add('active');
        }

        function hideSessionModal() {
            sessionModal.classList.remove('active');
        }

        function loadSessions() {
            // Get all session IDs from localStorage
            const sessionKeys = Object.keys(localStorage).filter(key => key.startsWith('chatSessionId_'));
            
            // Clear current session list
            sessionList.innerHTML = '';
            
            // Add current session
            const currentSessionItem = createSessionItem(sessionId, true);
            sessionList.appendChild(currentSessionItem);
            
            // Add other sessions
            sessionKeys.forEach(key => {
                const otherSessionId = key.replace('chatSessionId_', '');
                if (otherSessionId !== sessionId) {
                    const sessionItem = createSessionItem(otherSessionId, false);
                    sessionList.appendChild(sessionItem);
                }
            });
        }

        function createSessionItem(sessionId, isActive) {
            const sessionItem = document.createElement('div');
            sessionItem.className = `session-item ${isActive ? 'active' : ''}`;
            
            const sessionInfo = document.createElement('div');
            sessionInfo.className = 'session-info';
            
            const sessionName = document.createElement('div');
            sessionName.className = 'session-name';
            sessionName.textContent = isActive ? 'Current Session' : `Session ${sessionId.substring(8, 16)}`;
            
            const sessionDate = document.createElement('div');
            sessionDate.className = 'session-date';
            sessionDate.textContent = new Date().toLocaleDateString();
            
            sessionInfo.appendChild(sessionName);
            sessionInfo.appendChild(sessionDate);
            
            const sessionActions = document.createElement('div');
            sessionActions.className = 'session-actions';
            
            if (!isActive) {
                const switchBtn = document.createElement('button');
                switchBtn.className = 'session-action-btn';
                switchBtn.innerHTML = '<i class="fas fa-exchange-alt"></i>';
                switchBtn.title = 'Switch to this session';
                switchBtn.addEventListener('click', () => switchToSession(sessionId));
                
                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'session-action-btn delete';
                deleteBtn.innerHTML = '<i class="fas fa-trash"></i>';
                deleteBtn.title = 'Delete this session';
                deleteBtn.addEventListener('click', () => deleteSession(sessionId));
                
                sessionActions.appendChild(switchBtn);
                sessionActions.appendChild(deleteBtn);
            }
            
            sessionItem.appendChild(sessionInfo);
            sessionItem.appendChild(sessionActions);
            
            return sessionItem;
        }

        function createNewSession() {
            // Save current session ID
            localStorage.setItem(`chatSessionId_${sessionId}`, 'true');
            
            // Generate new session ID
            sessionId = generateSessionId();
            localStorage.setItem('chatSessionId', sessionId);
            
            // Clear chat UI
            chatMessages.innerHTML = `
                <div class="welcome-message">
                    <i class="fas fa-comments"></i>
                    <h2>Welcome to Claude 4.5 Assistant</h2>
                    <p>I'm here to help you with any questions or tasks you have. Just type your message below and press send.</p>
                </div>
            `;
            welcomeMessageRemoved = false;
            
            // Close modal
            hideSessionModal();
            
            // Show notification
            showNotification('New session created');
        }

        function switchToSession(newSessionId) {
            // Save current session ID
            localStorage.setItem(`chatSessionId_${sessionId}`, 'true');
            
            // Switch to new session
            sessionId = newSessionId;
            localStorage.setItem('chatSessionId', sessionId);
            
            // Clear chat UI
            chatMessages.innerHTML = `
                <div class="welcome-message">
                    <i class="fas fa-comments"></i>
                    <h2>Welcome to Claude 4.5 Assistant</h2>
                    <p>I'm here to help you with any questions or tasks you have. Just type your message below and press send.</p>
                </div>
            `;
            welcomeMessageRemoved = false;
            
            // Close modal
            hideSessionModal();
            
            // Show notification
            showNotification('Switched to session');
        }

        function deleteSession(sessionIdToDelete) {
            if (confirm('Are you sure you want to delete this session?')) {
                // Remove from localStorage
                localStorage.removeItem(`chatSessionId_${sessionIdToDelete}`);
                
                // If it's the current session, create a new one
                if (sessionIdToDelete === sessionId) {
                    createNewSession();
                } else {
                    // Refresh the session list
                    loadSessions();
                }
                
                // Show notification
                showNotification('Session deleted');
            }
        }

        // Show notification
        function showNotification(message) {
            const notification = document.createElement('div');
            notification.style.position = 'fixed';
            notification.style.bottom = '20px';
            notification.style.left = '50%';
            notification.style.transform = 'translateX(-50%)';
            notification.style.background = 'var(--bg-tertiary)';
            notification.style.color = 'var(--text-color)';
            notification.style.padding = '12px 24px';
            notification.style.borderRadius = '8px';
            notification.style.boxShadow = 'var(--shadow-lg)';
            notification.style.zIndex = '1000';
            notification.style.opacity = '0';
            notification.style.transition = 'opacity 0.3s ease';
            notification.textContent = message;
            
            document.body.appendChild(notification);
            
            // Fade in
            setTimeout(() => {
                notification.style.opacity = '1';
            }, 10);
            
            // Fade out and remove
            setTimeout(() => {
                notification.style.opacity = '0';
                setTimeout(() => {
                    document.body.removeChild(notification);
                }, 300);
            }, 3000);
        }

        // Focus input on load
        window.addEventListener('load', () => {
            userInput.focus();
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    logger.info("Index page accessed")
    return render_template_string(HTML_PAGE)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()
        session_id = request.headers.get("X-Session-ID")
        
        logger.info(f"Received message from session {session_id}: {len(user_input)} characters")
        
        if not user_input:
            logger.warning("Empty message received")
            return jsonify({"error": "Empty message"}), 400
        
        # Get or create conversation for this session
        if session_id not in conversations:
            conversations[session_id] = [
                {"role": "system", "content": """You are a helpful AI assistant with access to tools. When using tools, follow these guidelines:

1. ALWAYS provide ALL required parameters for tool calls.
2. Tool calls must be in the following format:
   {
     "name": "tool_name",
     "parameters": {
       "param1": "value1",
       "param2": "value2"
     }
   }

3. For the get_weather tool, you MUST provide:
   - name: "get_weather"
   - parameters: {"city": "City Name"} (city is required)

4. For the calculate tool, you MUST provide:
   - name: "calculate"
   - parameters: {"expression": "math expression"} (expression is required)

5. For the search_web tool, you MUST provide:
   - name: "search_web"
   - parameters: {"query": "search terms"} (query is required)

6. If a tool call fails, analyze the error message and try again with correct parameters.
7. Always explain what you're doing when using a tool."""}
            ]
            logger.info(f"Created new conversation for session {session_id}")
        
        conversation = conversations[session_id]
        conversation.append({"role": "user", "content": user_input})
        
        # Create the payload for the streaming request
        payload = {
            "model": MODEL,
            "messages": conversation,
            "stream": True,
            "max_tokens": 64000,  # API max limit is 64k
            "top_p": 0.9
        }
        
        # Add tools if the model supports them
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        
        logger.info(f"Sending streaming request to AI model for session {session_id}")
        logger.debug(f"Request payload: {json.dumps(payload)}")

        def generate():
            full_reply = ""
            chunk_count = 0
            tool_calls = []
            
            try:
                # Directly make the streaming request without a test request
                with requests.post(INFERENCE_URL, headers=headers, json=payload, stream=True, timeout=120) as response:
                    if response.status_code != 200:
                        logger.error(f"AI API returned status code {response.status_code}")
                        logger.error(f"Response headers: {response.headers}")
                        logger.error(f"Response text: {response.text[:500]}")  # Log first 500 chars of response
                        yield f"data: {json.dumps({'error': f'AI API returned status code {response.status_code}: {response.text[:200]}'})}\n\n"
                        return
                    
                    logger.info(f"Received response from AI API for session {session_id}")
                    client = sseclient.SSEClient(response)
                    
                    for event in client.events():
                        logger.debug(f"Received event: {event}")
                        if event.data.strip() == "[DONE]":
                            logger.info(f"Stream completed for session {session_id}, total chunks: {chunk_count}, total length: {len(full_reply)}")
                            break
                        try:
                            data = json.loads(event.data)
                            logger.debug(f"Parsed data: {data}")
                            
                            # Check for tool calls
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                
                                # Handle tool calls
                                if "tool_calls" in delta:
                                    for tool_call in delta["tool_calls"]:
                                        if "id" in tool_call:
                                            # New tool call
                                            tool_calls.append({
                                                "id": tool_call["id"],
                                                "type": tool_call.get("type", "function"),
                                                "function": {
                                                    "name": tool_call.get("function", {}).get("name", ""),
                                                    "arguments": tool_call.get("function", {}).get("arguments", "")
                                                }
                                            })
                                        elif "function" in tool_call:
                                            # Update existing tool call
                                            for existing_call in tool_calls:
                                                if existing_call["id"] == tool_call.get("id", ""):
                                                    if "name" in tool_call["function"]:
                                                        existing_call["function"]["name"] = tool_call["function"]["name"]
                                                    if "arguments" in tool_call["function"]:
                                                        existing_call["function"]["arguments"] += tool_call["function"]["arguments"]
                                
                                # Handle regular content
                                elif "content" in delta:
                                    content = delta["content"]
                                    if content:
                                        full_reply += content
                                        chunk_count += 1
                                        logger.debug(f"Yielding delta: {content}")
                                        # Format as SSE for proper frontend parsing
                                        yield f"data: {json.dumps(data)}\n\n"
                        except Exception as e:
                            logger.error(f"Error processing chunk: {str(e)}")
                            pass
                    
                    # Process tool calls if any
                    if tool_calls:
                        logger.info(f"Processing {len(tool_calls)} tool calls")
                        tool_results = []
                        
                        for tool_call in tool_calls:
                            tool_name = tool_call["function"]["name"]
                            # Parse arguments properly
                            try:
                                tool_args = json.loads(tool_call["function"]["arguments"]) if tool_call["function"]["arguments"] else {}
                            except json.JSONDecodeError:
                                # If arguments are not valid JSON, create an empty dict
                                tool_args = {}
                            
                            logger.info(f"Executing tool {tool_name} with args {tool_args}")
                            tool_result = execute_tool(tool_name, tool_args)
                            
                            # Check if tool result is an error
                            if isinstance(tool_result, dict) and "error" in tool_result:
                                logger.error(f"Tool execution error: {tool_result['error']}")
                                # Add error message to conversation for AI to correct
                                error_message = f"Error in tool call: {tool_result['error']}. "
                                if "example" in tool_result:
                                    error_message += f"Example of correct usage: {json.dumps(tool_result['example'])}"
                                
                                # Add a message to the conversation with the error and example
                                conversation.append({
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [tool_call]
                                })
                                conversation.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "name": tool_name,
                                    "content": error_message
                                })
                                
                                # Add a user message with instructions on how to fix the error
                                conversation.append({
                                    "role": "user",
                                    "content": f"You made an error in your tool call. {error_message}. Please try again with the correct parameters."
                                })
                                
                                # Yield the error message to the user
                                yield f"data: {json.dumps({'choices': [{'delta': {'content': f'I made an error in my tool call: {error_message}. Let me try again with the correct parameters.'}}]})}\n\n"
                            else:
                                tool_results.append({
                                    "tool_call_id": tool_call["id"],
                                    "output": tool_result
                                })
                                
                                # Add tool call and result to conversation
                                conversation.append({
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [tool_call]
                                })
                                conversation.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "name": tool_name,
                                    "content": tool_result
                                })
                        
                        # Send a new request with tool results
                        follow_up_payload = {
                            "model": MODEL,
                            "messages": conversation,
                            "stream": True,
                            "max_tokens": 64000,  # API max limit is 64k
                            "tools": tools,
                            "tool_choice": "auto",
                            "top_p": 0.9
                        }
                        
                        with requests.post(INFERENCE_URL, headers=headers, json=follow_up_payload, stream=True, timeout=120) as follow_up_response:
                            if follow_up_response.status_code != 200:
                                logger.error(f"Follow-up AI API returned status code {follow_up_response.status_code}")
                                yield f"data: {json.dumps({'error': f'Follow-up AI API returned status code {follow_up_response.status_code}'})}\n\n"
                                return
                            
                            follow_up_client = sseclient.SSEClient(follow_up_response)
                            for event in follow_up_client.events():
                                if event.data.strip() == "[DONE]":
                                    break
                                try:
                                    data = json.loads(event.data)
                                    delta = data["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        full_reply += delta
                                        chunk_count += 1
                                        logger.debug(f"Yielding follow-up delta: {delta}")
                                        yield f"data: {json.dumps(data)}\n\n"
                                except Exception as e:
                                    logger.error(f"Error processing follow-up chunk: {str(e)}")
                                    pass
                    
                    # Save the complete response to conversation
                    # Only add content if there's actual content, not just tool calls
                    if full_reply:
                        conversation.append({"role": "assistant", "content": full_reply})
                    logger.info(f"Saved response to conversation for session {session_id}")
            except requests.exceptions.Timeout:
                logger.error(f"Request timeout for session {session_id}")
                yield f"data: {json.dumps({'error': 'Request timeout. The response is taking too long to generate.'})}\n\n"
            except requests.exceptions.RequestException as e:
                logger.error(f"Request exception for session {session_id}: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            except Exception as e:
                logger.error(f"Unexpected error in generate() for session {session_id}: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(generate(), content_type="text/event-stream")
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/clear-chat", methods=["POST"])
def clear_chat():
    try:
        session_id = request.headers.get("X-Session-ID")
        logger.info(f"Clearing chat for session {session_id}")
        
        if session_id in conversations:
            conversations[session_id] = [
                {"role": "system", "content": """You are a helpful AI assistant with access to tools. When using tools, follow these guidelines:

1. ALWAYS provide ALL required parameters for tool calls.
2. Tool calls must be in the following format:
   {
     "name": "tool_name",
     "parameters": {
       "param1": "value1",
       "param2": "value2"
     }
   }

3. For the get_weather tool, you MUST provide:
   - name: "get_weather"
   - parameters: {"city": "City Name"} (city is required)

4. For the calculate tool, you MUST provide:
   - name: "calculate"
   - parameters: {"expression": "math expression"} (expression is required)

5. For the search_web tool, you MUST provide:
   - name: "search_web"
   - parameters: {"query": "search terms"} (query is required)

6. If a tool call fails, analyze the error message and try again with correct parameters.
7. Always explain what you're doing when using a tool."""}
            ]
            logger.info(f"Reset conversation for session {session_id}")
        
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Error in clear-chat endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ================== MAIN ==================
if __name__ == "__main__":
    from waitress import serve
    import os

    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting production server on port {port}")
    print(f"\n[Web UI] Running on http://0.0.0.0:{port}\n")
    serve(app, host="0.0.0.0", port=port, threads=8)
