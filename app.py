"""
Kira Web - AI Chat Assistant
=============================
Multi-model chat with voice support.
Supports: Claude, GPT, Gemini, DeepSeek, Llama
"""

import os
import json
import uuid
import base64
import tempfile
from datetime import datetime

import numpy as np
import soundfile as sf
try:
    import whisper as whisper
except ImportError:
    whisper = None
import anthropic
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, Response, stream_with_context
from dotenv import load_dotenv
from functools import wraps
from supabase import create_client as _sb_create_client

load_dotenv()

# ==========================================================================
# TRANSLATIONS  (ui_lang: en | de)
# ==========================================================================
TRANSLATIONS = {
    'en': {
        'new_chat': 'New Chat', 'audio_chat': 'Audio Conversation',
        'audio_subtitle': 'AI Voice Assistant',
        'tap_to_speak': 'Tap to speak', 'listening': 'Listening...',
        'thinking': 'Thinking...', 'kira_speaking': 'Kira is speaking...',
        'click_to_start': 'Click the orb to start recording',
        'click_to_stop': 'Click again to stop',
        'processing_msg': 'Processing your message...',
        'playing_resp': 'Playing response...',
        'start_voice': 'Start a voice conversation',
        'tap_orb_hint': 'Tap the orb above and start speaking',
        'clear': 'Clear', 'mic_denied': 'Microphone access denied.',
        'conn_error': 'Connection error.', 'space_hint': 'Press Space to start / stop',
        'settings': 'Settings', 'free': 'Free', 'share': 'Share',
        'cancel': 'Cancel', 'confirm': 'Confirm', 'select_model': 'Select Model',
        # profile
        'profile_title': 'Profile & Settings',
        'profile_subtitle': 'Manage your account and customize Kira.',
        'profile_section': 'Profile', 'profile_picture': 'Profile picture',
        'profile_picture_hint': 'JPG or PNG, max. 2 MB',
        'upload': 'Upload', 'remove': 'Remove',
        'first_name': 'First name', 'last_name': 'Last name',
        'username_label': 'Username',
        'username_hint': 'The username cannot be changed.',
        'save_profile': 'Save profile',
        'security_section': 'Security', 'change_password': 'Change password',
        'current_password': 'Current password',
        'new_password': 'New password', 'confirm_password': 'Confirm password',
        'settings_section': 'Settings',
        'appearance': 'Appearance', 'appearance_hint': 'Choose between light and dark mode.',
        'theme_system': 'System', 'theme_light': 'Light', 'theme_dark': 'Dark',
        'ai_lang_label': 'AI Language', 'ai_lang_hint': 'In which language should Kira respond?',
        'ui_lang_label': 'Website Language',
        'ui_lang_hint': 'Language of the user interface (excluding AI).',
        'context_memory_label': 'Context Memory',
        'context_memory_hint': 'Allows Kira to remember previous conversations for better responses.',
        'account_section': 'Account', 'logout': 'Sign out',
        'logout_hint': 'Safely sign out from this device.',
        # js strings
        'pw_fill_all': 'Please fill in all fields.',
        'pw_min_chars': 'Min. 8 characters required.',
        'pw_mismatch': 'Passwords do not match.',
        'saved': 'Settings saved.', 'profile_saved': 'Profile saved.',
        'pw_changed': 'Password changed successfully.',
        'avatar_saved': 'Profile picture saved.',
        'avatar_removed': 'Profile picture removed.',
        'avatar_too_big': 'Image too large. Max. 2 MB.',
        'pw_weak': 'Weak', 'pw_medium': 'Medium', 'pw_good': 'Good', 'pw_strong': 'Very strong',
        'first_name_ph': 'Your first name', 'last_name_ph': 'Your last name',
        'pw_min_ph': 'Min. 8 characters',
    },
    'de': {
        'new_chat': 'Neuer Chat', 'audio_chat': 'Audio-Konversation',
        'audio_subtitle': 'KI-Sprachassistent',
        'tap_to_speak': 'Tippen zum Sprechen', 'listening': 'Höre zu...',
        'thinking': 'Denke nach...', 'kira_speaking': 'Kira spricht...',
        'click_to_start': 'Klicke auf den Orb um aufzunehmen',
        'click_to_stop': 'Erneut klicken zum Stoppen',
        'processing_msg': 'Nachricht wird verarbeitet...',
        'playing_resp': 'Antwort wird abgespielt...',
        'start_voice': 'Starte ein Gespräch',
        'tap_orb_hint': 'Tippe auf den Orb oben und sprich',
        'clear': 'Leeren', 'mic_denied': 'Mikrofonzugriff verweigert.',
        'conn_error': 'Verbindungsfehler.', 'space_hint': 'Leertaste zum Starten / Stoppen',
        'settings': 'Einstellungen', 'free': 'Kostenlos', 'share': 'Teilen',
        'cancel': 'Abbrechen', 'confirm': 'Bestätigen', 'select_model': 'Modell auswählen',
        # profile
        'profile_title': 'Profil & Einstellungen',
        'profile_subtitle': 'Verwalte dein Konto und passe Kira an.',
        'profile_section': 'Profil', 'profile_picture': 'Profilbild',
        'profile_picture_hint': 'JPG oder PNG, max. 2 MB',
        'upload': 'Hochladen', 'remove': 'Entfernen',
        'first_name': 'Vorname', 'last_name': 'Nachname',
        'username_label': 'Benutzername',
        'username_hint': 'Der Benutzername kann nicht geändert werden.',
        'save_profile': 'Profil speichern',
        'security_section': 'Sicherheit', 'change_password': 'Passwort ändern',
        'current_password': 'Aktuelles Passwort',
        'new_password': 'Neues Passwort', 'confirm_password': 'Passwort bestätigen',
        'settings_section': 'Einstellungen',
        'appearance': 'Erscheinungsbild', 'appearance_hint': 'Wähle zwischen hellem und dunklem Modus.',
        'theme_system': 'System', 'theme_light': 'Hell', 'theme_dark': 'Dunkel',
        'ai_lang_label': 'Sprache der KI', 'ai_lang_hint': 'In welcher Sprache soll Kira antworten?',
        'ui_lang_label': 'Sprache der Website',
        'ui_lang_hint': 'Sprache der Benutzeroberfläche (ohne KI).',
        'context_memory_label': 'Kontext-Erinnerung',
        'context_memory_hint': 'Erlaubt Kira, sich an frühere Konversationen zu erinnern, um bessere Antworten zu geben.',
        'account_section': 'Konto', 'logout': 'Abmelden',
        'logout_hint': 'Sicher von diesem Gerät abmelden.',
        # js strings
        'pw_fill_all': 'Bitte alle Felder ausfüllen.',
        'pw_min_chars': 'Min. 8 Zeichen erforderlich.',
        'pw_mismatch': 'Passwörter stimmen nicht überein.',
        'saved': 'Einstellung gespeichert.', 'profile_saved': 'Profil gespeichert.',
        'pw_changed': 'Passwort erfolgreich geändert.',
        'avatar_saved': 'Profilbild gespeichert.',
        'avatar_removed': 'Profilbild entfernt.',
        'avatar_too_big': 'Bild zu groß. Max. 2 MB.',
        'pw_weak': 'Schwach', 'pw_medium': 'Mittel', 'pw_good': 'Gut', 'pw_strong': 'Sehr stark',
        'first_name_ph': 'Dein Vorname', 'last_name_ph': 'Dein Nachname',
        'pw_min_ph': 'Min. 8 Zeichen',
    },
}

AI_LANG_INSTRUCTIONS = {
    'en': 'Always respond in English, regardless of the language the user writes in.',
    'de': 'Antworte immer auf Deutsch, egal in welcher Sprache der Nutzer schreibt.',
}

DEFAULT_SETTINGS = {
    'ai_lang': 'en',
    'ui_lang': 'en',
    'context_memory': True,
    'firstname': '',
    'lastname': '',
}

# ==========================================================================
# FLASK APP
# ==========================================================================
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex())

# ==========================================================================
# SUPABASE CLIENT
# ==========================================================================
_sb_url = os.getenv("SUPABASE_URL", "")
_sb_key = os.getenv("SUPABASE_SERVICE_KEY", "")
supabase_client = None
if _sb_url and _sb_key and not _sb_url.startswith("https://your-project"):
    try:
        supabase_client = _sb_create_client(_sb_url, _sb_key)
        print("Supabase connected!")
    except Exception as e:
        print(f"Supabase init failed: {e}")
else:
    print("WARNING: Supabase not configured — add SUPABASE_URL and SUPABASE_SERVICE_KEY to .env")


# ==========================================================================
# USERS
# ==========================================================================
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")


def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def get_user_settings(username=None):
    if username is None:
        username = session.get("username", "")
    users = load_users()
    for u in users:
        if u["username"] == username:
            return {**DEFAULT_SETTINGS, **u.get("settings", {})}
    return dict(DEFAULT_SETTINGS)


def update_user_settings(new_settings, username=None):
    if username is None:
        username = session.get("username", "")
    users = load_users()
    for u in users:
        if u["username"] == username:
            current = {**DEFAULT_SETTINGS, **u.get("settings", {})}
            current.update(new_settings)
            u["settings"] = current
            save_users(users)
            # Sync relevant keys to session
            for k in ("ai_lang", "ui_lang", "context_memory"):
                if k in new_settings:
                    session[k] = new_settings[k]
            return current
    return dict(DEFAULT_SETTINGS)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def build_system_prompt(extra=''):
    """Build a system prompt with the current user's AI language injected."""
    ai_lang = session.get('ai_lang', 'en')
    lang_instr = AI_LANG_INSTRUCTIONS.get(ai_lang, AI_LANG_INSTRUCTIONS['en'])
    base = (
        "You are Kira, a friendly and helpful AI assistant. "
        "You answer clearly and concisely. Be helpful, natural, and engaging. "
        + lang_instr
    )
    if extra:
        base += ' ' + extra
    return base


@app.context_processor
def inject_globals():
    lang = session.get('ui_lang', 'en')
    T = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
    display_name = None
    display_initial = None
    if session.get('logged_in'):
        s = get_user_settings()
        fn = s.get('firstname', '').strip()
        ln = s.get('lastname', '').strip()
        full = f"{fn} {ln}".strip()
        if full:
            display_name = full
            display_initial = full[0].upper()
    if not display_name:
        display_name = session.get('username', 'User')
        display_initial = display_name[0].upper() if display_name else 'U'
    return {'T': T, 'ui_lang': lang, 'display_name': display_name, 'display_initial': display_initial}


# ==========================================================================
# WHISPER (Speech-to-Text)
# ==========================================================================
# Groq STT (fast cloud Whisper) — falls back to local if no key
groq_stt_client = None
try:
    from groq import Groq as _Groq
    if os.getenv("GROQ_API_KEY"):
        groq_stt_client = _Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("[STT] Groq Whisper ready (fast mode)")
    else:
        print("[STT] No GROQ_API_KEY — using local Whisper")
except Exception as e:
    print(f"[STT] Groq unavailable: {e}")

# Local Whisper fallback
print("Loading local Whisper model (tiny)...")
whisper_model = whisper.load_model("tiny")
print("Whisper ready!")

# ==========================================================================
# AI CLIENTS
# ==========================================================================

# Anthropic (Claude)
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Google Gemini
gemini_configured = False
gemini_client = None
try:
    from google import genai as google_genai
    if os.getenv("GOOGLE_API_KEY"):
        gemini_client = google_genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        gemini_configured = True
except Exception:
    pass

# Llama via Together.ai
together_client = None
try:
    from openai import OpenAI as _OpenAI
    if os.getenv("TOGETHER_API_KEY"):
        together_client = _OpenAI(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1"
        )
except ImportError:
    pass

# Text-to-Speech (edge-tts — Microsoft neural voices, free)
import io, asyncio

EDGE_VOICES = {
    "de": "de-DE-SeraphinaMultilingualNeural",
    "en": "en-US-AvaMultilingualNeural",
}

try:
    import edge_tts as _edge_tts
    _edge_available = True
    print("[TTS] edge-tts ready")
except Exception:
    _edge_available = False
    print("[TTS] edge-tts not available")

import re as _re, unicodedata as _ud

def _clean_transcript(text):
    """Keep only letters, digits, spaces and common punctuation. Strip emojis/symbols."""
    out = []
    for ch in text:
        cat = _ud.category(ch)
        # Allow Unicode letters (incl. ä ö ü ß etc.) and numbers
        if cat.startswith('L') or cat.startswith('N'):
            out.append(ch)
        # Allow whitespace
        elif cat in ('Zs', 'Cc') or ch in (' ', '\t', '\n'):
            out.append(' ')
        # Allow common punctuation and symbols explicitly
        elif ch in '.,!?;:\'"()[]{}#+*-=/%&@€$':
            out.append(ch)
        # Everything else (emojis, icons, special Unicode) → drop
    text = ''.join(out)
    text = _re.sub(r' {2,}', ' ', text).strip()
    return text

_MATH_DE = [
    (r'²', ' hoch zwei'), (r'³', ' hoch drei'), (r'⁴', ' hoch vier'),
    (r'√', 'Wurzel aus '), (r'π', 'Pi'), (r'∞', 'unendlich'),
    (r'≈', ' ungefähr gleich '), (r'≠', ' ungleich '), (r'≤', ' kleiner gleich '),
    (r'≥', ' größer gleich '), (r'×', ' mal '), (r'÷', ' geteilt durch '),
    (r'\^(\d+)', r' hoch \1'),
]
_MATH_EN = [
    (r'²', ' squared'), (r'³', ' cubed'), (r'⁴', ' to the power of 4'),
    (r'√', 'square root of '), (r'π', 'pi'), (r'∞', 'infinity'),
    (r'≈', ' approximately equals '), (r'≠', ' not equal to '), (r'≤', ' less than or equal to '),
    (r'≥', ' greater than or equal to '), (r'×', ' times '), (r'÷', ' divided by '),
    (r'\^(\d+)', r' to the power of \1'),
]

def _normalize_for_tts(text, lang="de"):
    """Convert math symbols to spoken words and strip markdown."""
    # Strip markdown first
    text = _re.sub(r'```[\s\S]*?```', '', text)
    text = _re.sub(r'`([^`]+)`', r'\1', text)
    text = _re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = _re.sub(r'\*(.+?)\*', r'\1', text)
    text = _re.sub(r'__(.+?)__', r'\1', text)
    text = _re.sub(r'_(.+?)_', r'\1', text)
    text = _re.sub(r'^#{1,6}\s+', '', text, flags=_re.MULTILINE)
    text = _re.sub(r'^\s*[-*+]\s+', '', text, flags=_re.MULTILINE)
    text = _re.sub(r'^\s*\d+\.\s+', '', text, flags=_re.MULTILINE)
    text = _re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove quotation marks (all variants) — they sound odd when read aloud
    text = text.replace('"', '').replace('"', '').replace('"', '').replace("'", '').replace('„', '').replace('«', '').replace('»', '')
    # Normalize math symbols to spoken words
    replacements = _MATH_DE if lang == "de" else _MATH_EN
    for pattern, repl in replacements:
        text = _re.sub(pattern, repl, text)
    return text.strip()

def _strip_markdown(text):
    return _normalize_for_tts(text, lang="en")

async def _edge_synthesize(text, voice):
    communicate = _edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()

import threading as _threading
_tts_loop_local = _threading.local()

def _get_tts_loop():
    if not getattr(_tts_loop_local, 'loop', None) or _tts_loop_local.loop.is_closed():
        _tts_loop_local.loop = asyncio.new_event_loop()
    return _tts_loop_local.loop

def text_to_speech(text, lang="en"):
    if not _edge_available:
        return None
    try:
        clean = _normalize_for_tts(text, lang).strip()
        if not clean:
            return None
        voice = EDGE_VOICES.get(lang, EDGE_VOICES["en"])
        return _get_tts_loop().run_until_complete(_edge_synthesize(clean, voice))
    except Exception as e:
        print(f"[TTS] edge-tts error: {e}")
        return None

# ==========================================================================
# MODEL MAPPING
# ==========================================================================
MODEL_MAP = {
    "claude-sonnet": {"provider": "claude", "model": "claude-sonnet-4-20250514"},
    "claude-opus": {"provider": "claude", "model": "claude-opus-4-20250514"},
    "gemini-2.5-pro": {"provider": "gemini", "model": "gemini-2.5-pro"},
    "gemini-2.5-flash": {"provider": "gemini", "model": "gemini-2.5-flash"},
    "llama-4": {"provider": "together", "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"},
}

SYSTEM_PROMPT = (
    "You are Kira, a friendly and helpful AI assistant. "
    "You answer clearly and concisely. Be helpful, natural, and engaging."
)



# ==========================================================================
# CONVERSATION STORAGE (Supabase)
# ==========================================================================

def get_or_create_conv(conv_id=None):
    """Fetch an existing conversation from Supabase or create a new one."""
    username = session.get("username", "anonymous")

    if conv_id and supabase_client:
        try:
            resp = supabase_client.table("conversations") \
                .select("*") \
                .eq("id", conv_id) \
                .eq("username", username) \
                .execute()
            if resp.data:
                row = resp.data[0]
                conv = {
                    "title": row["title"],
                    "messages": row["messages"] or [],
                    "model": row["model"],
                    "created": row["created"],
                }
                return conv_id, conv
        except Exception as e:
            print(f"Supabase fetch error: {e}")

    new_id = str(uuid.uuid4())[:8]
    conv = {
        "title": "New Chat",
        "messages": [],
        "model": "claude-sonnet",
        "created": datetime.now().isoformat(),
    }
    if supabase_client:
        try:
            supabase_client.table("conversations").insert({
                "id": new_id,
                "username": username,
                "title": conv["title"],
                "messages": conv["messages"],
                "model": conv["model"],
                "created": conv["created"],
            }).execute()
        except Exception as e:
            print(f"Supabase insert error: {e}")
    return new_id, conv


def save_conv(conv_id, conv, username=None):
    """Persist conversation changes (messages, title, model) to Supabase."""
    if not supabase_client:
        return
    if username is None:
        username = session.get("username", "anonymous")
    try:
        supabase_client.table("conversations").update({
            "title": conv["title"],
            "messages": conv["messages"],
            "model": conv["model"],
        }).eq("id", conv_id).eq("username", username).execute()
    except Exception as e:
        print(f"Supabase save error: {e}")


def generate_title(text):
    """Generate a short title from the first message."""
    words = text.strip().split()
    if len(words) <= 6:
        return text.strip()
    return " ".join(words[:6]) + "..."


# ==========================================================================
# WEB SEARCH (for Deep Research)
# ==========================================================================
def web_search(query, num_results=8):
    """Search the web using DuckDuckGo and return results."""
    try:
        from ddgs import DDGS
        ddgs = DDGS()
        results = []
        for r in ddgs.text(query, max_results=num_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []


def deep_research(query, model_id):
    """Perform deep research: search web, then synthesize with AI."""
    # Search for multiple angles
    results = web_search(query, num_results=10)

    if not results:
        return call_ai(model_id, [{"role": "user", "content": query}]), []

    # Build context from search results
    sources_text = ""
    sources = []
    for i, r in enumerate(results, 1):
        sources_text += f"\n[{i}] {r['title']}\nURL: {r['url']}\n{r['snippet']}\n"
        sources.append({"title": r["title"], "url": r["url"]})

    research_prompt = (
        f"The user asked: {query}\n\n"
        f"I searched the web and found these results:\n{sources_text}\n\n"
        "IMPORTANT: Use the search results above to answer. Do NOT say you cannot access the internet. "
        "Synthesize information from these sources. Cite with [1], [2], etc."
    )

    ai_lang = session.get('ai_lang', 'en')
    lang_instr = AI_LANG_INSTRUCTIONS.get(ai_lang, AI_LANG_INSTRUCTIONS['en'])
    system = (
        "You are Kira, an AI research assistant with web search capabilities. "
        "Search results are provided. NEVER say you cannot access the internet. "
        "Use the provided results to give detailed answers. Cite sources with [1], [2], etc. "
        + lang_instr
    )

    # Override system prompt for research mode
    config = MODEL_MAP.get(model_id, MODEL_MAP["claude-sonnet"])
    provider = config["provider"]
    model = config["model"]

    try:
        if provider == "claude":
            response = claude_client.messages.create(
                model=model, max_tokens=8192, system=system,
                messages=[{"role": "user", "content": research_prompt}],
            )
            return response.content[0].text, sources
        else:
            return call_ai(model_id, [{"role": "user", "content": research_prompt}]), sources
    except Exception as e:
        return f"Research error: {str(e)}", sources


def generate_image(prompt):
    """Generate an image URL using Pollinations.ai (free, no API key needed)."""
    import urllib.parse
    encoded = urllib.parse.quote(prompt)
    seed = hash(prompt) % 100000
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&seed={seed}"
    return url, None


# ==========================================================================
# AI CALL ROUTING
# ==========================================================================
def call_ai(model_id, messages, think_mode=False):
    """Route to the correct AI provider and return the response text."""
    config = MODEL_MAP.get(model_id)
    if not config:
        return "Unknown model selected."

    provider = config["provider"]
    model = config["model"]
    max_tok = 16384 if think_mode else 4096
    extra = ""
    if think_mode:
        extra = (
            "The user wants a very thorough, detailed answer. "
            "Take your time to think step by step. Consider multiple angles. "
            "Provide a comprehensive, well-structured response with deep analysis."
        )
    system = build_system_prompt(extra)

    try:
        if provider == "claude":
            response = claude_client.messages.create(
                model=model,
                max_tokens=max_tok,
                system=system,
                messages=messages,
            )
            return response.content[0].text

        elif provider == "gemini":
            if not gemini_configured or not gemini_client:
                return "Google API key not configured. Add GOOGLE_API_KEY to your .env file."
            contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages]
            response = gemini_client.models.generate_content(
                model=model,
                contents=contents,
                config={"system_instruction": system, "max_output_tokens": max_tok},
            )
            return response.text

        elif provider == "together":
            if not together_client:
                return "Together API key not configured. Add TOGETHER_API_KEY to your .env file."
            sys_msgs = [{"role": "system", "content": system}]
            response = together_client.chat.completions.create(
                model=model,
                messages=sys_msgs + messages,
                max_tokens=max_tok,
            )
            return response.choices[0].message.content

        else:
            return "Provider not supported."

    except Exception as e:
        print(f"AI Error ({provider}/{model}): {e}")
        return f"Error from {provider}: {str(e)}"


# ==========================================================================
# ROUTES
# ==========================================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        users = load_users()
        for user in users:
            if user["username"] == username and user["password"] == password:
                session["logged_in"] = True
                session["username"] = username
                # Load user settings into session
                s = {**DEFAULT_SETTINGS, **user.get("settings", {})}
                session["ai_lang"] = s["ai_lang"]
                session["ui_lang"] = s["ui_lang"]
                session["context_memory"] = s["context_memory"]
                return redirect(url_for("index"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/audio")
@login_required
def audio():
    return render_template("audio.html")


@app.route("/profile")
@login_required
def profile():
    user_settings = get_user_settings()
    return render_template("profile.html", user_settings=user_settings)


@app.route("/api/settings", methods=["GET"])
@login_required
def api_get_settings():
    return jsonify(get_user_settings())


@app.route("/api/settings", methods=["POST"])
@login_required
def api_save_settings():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    allowed = {"ai_lang", "ui_lang", "context_memory", "firstname", "lastname"}
    filtered = {k: v for k, v in data.items() if k in allowed}
    # Validate language values
    if "ai_lang" in filtered and filtered["ai_lang"] not in ("en", "de"):
        return jsonify({"error": "Invalid ai_lang"}), 400
    if "ui_lang" in filtered and filtered["ui_lang"] not in ("en", "de"):
        return jsonify({"error": "Invalid ui_lang"}), 400
    updated = update_user_settings(filtered)
    return jsonify({"status": "ok", "settings": updated})


@app.route("/chat/<conv_id>")
@login_required
def chat_page(conv_id):
    username = session.get("username", "anonymous")
    if supabase_client:
        try:
            resp = supabase_client.table("conversations") \
                .select("id") \
                .eq("id", conv_id) \
                .eq("username", username) \
                .execute()
            if not resp.data:
                return redirect(url_for("index"))
        except Exception:
            return redirect(url_for("index"))
    return render_template("chat.html", conversation_id=conv_id)




# --- Text Chat API ---
@app.route("/api/text-chat", methods=["POST"])
@login_required
def text_chat():
    """Handle text-based chat messages."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    user_text = data.get("message", "").strip()
    model_id = data.get("model", "claude-sonnet")
    conv_id = data.get("conversation_id")
    mode = data.get("mode", "normal")  # normal, research, image, think

    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    # Get or create conversation
    conv_id, conv = get_or_create_conv(conv_id)
    conv["model"] = model_id

    if not conv["messages"]:
        conv["title"] = generate_title(user_text)

    # Add user message
    conv["messages"].append({"role": "user", "content": user_text})

    sources = None
    image_url = None
    context_memory = session.get("context_memory", True)
    ai_messages = conv["messages"] if context_memory else [conv["messages"][-1]]

    if mode == "research":
        assistant_text, sources = deep_research(user_text, model_id)
    elif mode == "image":
        img_url, img_err = generate_image(user_text)
        if img_url:
            image_url = img_url
            assistant_text = f"Here's the image I generated for: \"{user_text}\""
        else:
            assistant_text = img_err or "Could not generate image."
    elif mode == "think":
        assistant_text = call_ai(model_id, ai_messages, think_mode=True)
    else:
        assistant_text = call_ai(model_id, ai_messages)

    conv["messages"].append({"role": "assistant", "content": assistant_text})
    save_conv(conv_id, conv)

    result = {
        "assistant_text": assistant_text,
        "conversation_id": conv_id,
        "title": conv["title"],
    }
    if sources:
        result["sources"] = sources
    if image_url:
        result["image_url"] = image_url

    return jsonify(result)


# --- Real Token Streaming ---

def stream_claude(model, messages, system, max_tokens=4096):
    """Yield tokens from Claude streaming API."""
    with claude_client.messages.stream(
        model=model, max_tokens=max_tokens, system=system, messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def stream_gemini(model, messages, system, max_tokens=4096):
    """Yield tokens from Gemini streaming API."""
    contents = [{"role": "user" if m["role"] == "user" else "model",
                 "parts": [{"text": m["content"]}]} for m in messages]
    response = gemini_client.models.generate_content_stream(
        model=model, contents=contents,
        config={"system_instruction": system, "max_output_tokens": max_tokens},
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


def stream_together(model, messages, system, max_tokens=4096):
    """Yield tokens from Together.ai streaming API."""
    sys_msgs = [{"role": "system", "content": system}]
    stream = together_client.chat.completions.create(
        model=model, messages=sys_msgs + messages, max_tokens=max_tokens, stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def stream_ai(model_id, messages, system=None, max_tokens=4096):
    """Route to the correct streaming provider."""
    if system is None:
        system = build_system_prompt()
    config = MODEL_MAP.get(model_id, MODEL_MAP["claude-sonnet"])
    provider = config["provider"]
    model = config["model"]

    if provider == "claude":
        yield from stream_claude(model, messages, system, max_tokens)
    elif provider == "gemini" and gemini_client:
        yield from stream_gemini(model, messages, system, max_tokens)
    elif provider == "together" and together_client:
        yield from stream_together(model, messages, system, max_tokens)
    else:
        # Fallback: non-streaming
        yield call_ai(model_id, messages)


# --- Streaming Text Chat API (SSE) ---
@app.route("/api/text-chat-stream", methods=["POST"])
@login_required
def text_chat_stream():
    """Stream chat response via Server-Sent Events with real token streaming."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    user_text = data.get("message", "").strip()
    model_id = data.get("model", "claude-sonnet")
    conv_id = data.get("conversation_id")
    mode = data.get("mode", "normal")

    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    conv_id, conv = get_or_create_conv(conv_id)
    conv["model"] = model_id

    if not conv["messages"]:
        conv["title"] = generate_title(user_text)

    conv["messages"].append({"role": "user", "content": user_text})

    context_memory = session.get("context_memory", True)
    ai_messages = conv["messages"] if context_memory else [conv["messages"][-1]]

    import time

    def sse(event, payload):
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

    def generate():
        sources = None
        image_url = None
        full_text = ""

        try:
            if mode == "research":
                # Phase 1: Web search with live site updates
                yield sse("status", {"text": "Searching the web...", "icon": "search"})
                search_results = web_search(user_text, num_results=10)

                if search_results:
                    sources = []
                    sources_text = ""
                    for i, r in enumerate(search_results):
                        domain = r["url"].split("/")[2] if len(r["url"].split("/")) > 2 else r["url"]
                        yield sse("status", {"text": f"Reading {domain}...", "icon": "globe", "url": r["url"]})
                        sources_text += f"\n[{i+1}] {r['title']}\nURL: {r['url']}\n{r['snippet']}\n"
                        sources.append({"title": r["title"], "url": r["url"]})
                        time.sleep(0.4)

                    yield sse("status", {"text": "Analyzing and writing response...", "icon": "think"})
                    time.sleep(0.3)

                    research_prompt = (
                        f"The user asked: {user_text}\n\n"
                        f"I searched the web and found these results:\n{sources_text}\n\n"
                        "IMPORTANT: You MUST use the web search results above to answer. "
                        "Do NOT say you cannot access the internet - the results are already provided above. "
                        "Synthesize the information from these sources into a comprehensive answer. "
                        "Cite sources using [1], [2], etc. Be thorough and detailed."
                    )
                    research_system = (
                        "You are Kira, an AI research assistant with web search capabilities. "
                        "You have already searched the web and the results are provided in the user message. "
                        "NEVER say you cannot access the internet. Use the provided search results to give "
                        "detailed, well-researched answers. Always cite your sources with [1], [2], etc."
                    )

                    # Phase 2: Stream AI response token by token
                    yield sse("stream_start", {})
                    for token in stream_ai(model_id, [{"role": "user", "content": research_prompt}],
                                           system=research_system, max_tokens=8192):
                        full_text += token
                        yield sse("chunk", {"text": token})
                else:
                    yield sse("status", {"text": "No results found, thinking...", "icon": "think"})
                    yield sse("stream_start", {})
                    for token in stream_ai(model_id, ai_messages):
                        full_text += token
                        yield sse("chunk", {"text": token})

            elif mode == "image":
                yield sse("status", {"text": "Generating image...", "icon": "image"})
                img_url, img_err = generate_image(user_text)
                if img_url:
                    image_url = img_url
                    full_text = f'Here\'s the image I generated for: "{user_text}"'
                else:
                    full_text = img_err or "Could not generate image."
                yield sse("stream_start", {})
                yield sse("chunk", {"text": full_text})

            elif mode == "think":
                yield sse("status", {"text": "Thinking deeply...", "icon": "think"})
                time.sleep(0.5)
                yield sse("status", {"text": "Considering multiple angles...", "icon": "think"})
                system_think = build_system_prompt(
                    "The user wants a very thorough, detailed answer. "
                    "Think step by step. Consider multiple angles. "
                    "Provide a comprehensive, well-structured response."
                )
                yield sse("stream_start", {})
                for token in stream_ai(model_id, ai_messages,
                                       system=system_think, max_tokens=16384):
                    full_text += token
                    yield sse("chunk", {"text": token})

            else:
                yield sse("status", {"text": "Thinking...", "icon": "think"})
                yield sse("stream_start", {})
                for token in stream_ai(model_id, ai_messages):
                    full_text += token
                    yield sse("chunk", {"text": token})

        except Exception as e:
            err_msg = str(e)
            if not full_text:
                yield sse("stream_start", {})
            yield sse("chunk", {"text": f"\n\nError: {err_msg}"})
            full_text += f"\n\nError: {err_msg}"

        # Save to conversation
        conv["messages"].append({"role": "assistant", "content": full_text})
        save_conv(conv_id, conv)

        # Send done with metadata
        done_data = {"conversation_id": conv_id, "title": conv["title"]}
        if sources:
            done_data["sources"] = sources
        if image_url:
            done_data["image_url"] = image_url
        yield sse("done", done_data)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- Voice Chat API ---
@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    """Handle voice-based chat (audio in, text + audio out)."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    model_id = request.form.get("model", "claude-sonnet")
    conv_id = request.form.get("conversation_id")

    audio_file = request.files["audio"]
    temp_input = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_file.save(temp_input.name)
    temp_input.close()

    try:
        # Speech-to-Text
        lang_hint = session.get("ai_lang", "de")

        if groq_stt_client:
            # Fast path: Groq cloud Whisper (~0.2s)
            with open(temp_input.name, "rb") as af:
                transcription = groq_stt_client.audio.transcriptions.create(
                    file=("audio.wav", af),
                    model="whisper-large-v3-turbo",
                    language=lang_hint,
                    response_format="text",
                )
            raw_text = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
        else:
            # Fallback: local Whisper
            audio_data, sample_rate = sf.read(temp_input.name)
            audio_data = audio_data.astype(np.float32)
            result = whisper_model.transcribe(
                audio_data, language=lang_hint, fp16=False,
                temperature=0, best_of=1, beam_size=5,
                condition_on_previous_text=False,
                no_speech_threshold=0.6, compression_ratio_threshold=2.4,
                logprob_threshold=-0.8,
            )
            segments = result.get("segments", [])
            good_parts = [
                seg["text"] for seg in segments
                if seg.get("avg_logprob", -999) > -0.7
                and seg.get("no_speech_prob", 1.0) < 0.4
            ]
            raw_text = " ".join(good_parts).strip() if good_parts else result["text"].strip()

        user_text = _clean_transcript(raw_text)
        if user_text:
            words = user_text.split()
            one_char_ratio = sum(1 for w in words if len(w) <= 1) / max(len(words), 1)
            unique_ratio   = len(set(words)) / max(len(words), 1)
            if one_char_ratio > 0.5 or (len(words) > 3 and unique_ratio < 0.35):
                user_text = ""

        if not user_text:
            return jsonify({"user_text": "", "assistant_text": "", "error": "Nothing recognized"})

        # Get or create conversation
        conv_id, conv = get_or_create_conv(conv_id)
        conv["model"] = model_id

        if not conv["messages"]:
            conv["title"] = generate_title(user_text)

        # Read session values before streaming (no Flask context in generator)
        lang         = session.get("ai_lang", session.get("ui_lang", "en"))
        username     = session.get("username", "anonymous")
        voice        = EDGE_VOICES.get(lang, EDGE_VOICES["en"])
        voice_system = build_system_prompt(
            "You are in a voice conversation. Rules you must follow strictly: "
            "1. Maximum 2 sentences per answer. "
            "2. NEVER ask a question or follow-up. "
            "3. NEVER include phonetic transcriptions, pronunciation guides, or text in quotes meant to show how something sounds. "
            "4. Just answer directly and stop. No filler phrases like 'Wie kann ich dir helfen?' or 'Ich bin bereit'."
        )

        conv["messages"].append({"role": "user", "content": user_text})

        def generate():
            # 1. Send transcript immediately
            yield f"event: transcript\ndata: {json.dumps({'text': user_text, 'conversation_id': conv_id, 'title': conv['title']})}\n\n"

            # 2. Stream Claude + synthesize sentences on the fly
            sentence_buf = ""
            full_text    = ""
            sentence_end = _re.compile(r'(?<=[.!?])\s+')

            for chunk in stream_ai(model_id, conv["messages"], system=voice_system):
                full_text    += chunk
                sentence_buf += chunk

                # Flush complete sentences to TTS immediately
                parts = sentence_end.split(sentence_buf)
                for sentence in parts[:-1]:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    audio_bytes = text_to_speech(sentence, lang)
                    payload = {"text": sentence}
                    if audio_bytes:
                        payload["audio"] = base64.b64encode(audio_bytes).decode("utf-8")
                    yield f"event: sentence\ndata: {json.dumps(payload)}\n\n"
                sentence_buf = parts[-1]

            # Flush remaining text
            if sentence_buf.strip():
                audio_bytes = text_to_speech(sentence_buf.strip(), lang)
                payload = {"text": sentence_buf.strip()}
                if audio_bytes:
                    payload["audio"] = base64.b64encode(audio_bytes).decode("utf-8")
                yield f"event: sentence\ndata: {json.dumps(payload)}\n\n"

            # 3. Save conversation
            conv["messages"].append({"role": "assistant", "content": full_text})
            save_conv(conv_id, conv, username)
            yield f"event: done\ndata: {json.dumps({'conversation_id': conv_id})}\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        os.unlink(temp_input.name)


# --- Conversation Management ---
@app.route("/api/conversations", methods=["GET"])
@login_required
def list_conversations():
    """Return all conversations for the current user."""
    username = session.get("username", "anonymous")
    if not supabase_client:
        return jsonify([])
    try:
        resp = supabase_client.table("conversations") \
            .select("id, title, model, created, messages") \
            .eq("username", username) \
            .order("created", desc=True) \
            .execute()
        result = []
        for row in (resp.data or []):
            result.append({
                "id": row["id"],
                "title": row["title"],
                "model": row["model"],
                "created": row["created"],
                "message_count": len(row.get("messages") or []),
            })
        return jsonify(result)
    except Exception as e:
        print(f"Supabase list error: {e}")
        return jsonify([])


@app.route("/api/conversations/<conv_id>", methods=["GET"])
@login_required
def get_conversation(conv_id):
    """Return a specific conversation with all messages."""
    username = session.get("username", "anonymous")
    if not supabase_client:
        return jsonify({"error": "Database not configured"}), 503
    try:
        resp = supabase_client.table("conversations") \
            .select("*") \
            .eq("id", conv_id) \
            .eq("username", username) \
            .execute()
        if not resp.data:
            return jsonify({"error": "Conversation not found"}), 404
        row = resp.data[0]
        return jsonify({
            "id": row["id"],
            "title": row["title"],
            "model": row["model"],
            "messages": row["messages"] or [],
            "created": row["created"],
        })
    except Exception as e:
        print(f"Supabase get error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
@login_required
def delete_conversation(conv_id):
    """Delete a conversation."""
    username = session.get("username", "anonymous")
    if supabase_client:
        try:
            supabase_client.table("conversations") \
                .delete() \
                .eq("id", conv_id) \
                .eq("username", username) \
                .execute()
        except Exception as e:
            print(f"Supabase delete error: {e}")
    return jsonify({"status": "ok"})


@app.route("/api/reset", methods=["POST"])
@login_required
def reset():
    """Reset current conversation (create new)."""
    return jsonify({"status": "ok"})


@app.route("/api/models", methods=["GET"])
@login_required
def get_models():
    """Return available models with live status check."""
    result = {}
    for model_id, config in MODEL_MAP.items():
        provider = config["provider"]
        model = config["model"]
        available = False
        reason = ""

        try:
            if provider == "claude":
                if not os.getenv("ANTHROPIC_API_KEY"):
                    reason = "API key missing"
                else:
                    claude_client.messages.create(
                        model=model, max_tokens=5, system="Reply with OK",
                        messages=[{"role": "user", "content": "hi"}],
                    )
                    available = True
            elif provider == "gemini":
                if not gemini_configured or not gemini_client:
                    reason = "API key missing"
                else:
                    gemini_client.models.generate_content(
                        model=model, contents="hi",
                        config={"max_output_tokens": 5},
                    )
                    available = True
            elif provider == "together":
                if not together_client:
                    reason = "API key missing"
                else:
                    together_client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": "hi"}],
                        max_tokens=5,
                    )
                    available = True
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                reason = "Rate limit reached"
            elif "402" in err or "credit" in err.lower():
                reason = "No credits"
            elif "401" in err or "auth" in err.lower():
                reason = "Invalid API key"
            else:
                reason = "Unavailable"

        result[model_id] = {"available": available, "provider": provider, "reason": reason}
    return jsonify(result)


# ==========================================================================
# SERVER START
# ==========================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nKira Web running on: http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
