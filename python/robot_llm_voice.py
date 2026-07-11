"""
MIRA brain — mic in, arm does stuff.

Flow is pretty simple:
  whisper hears you -> groq figures out the intent -> we either talk back
  and/or yell at the Arduino / vision loop.

Arduino side talks in plain text commands (TRACK, GRAB, WAVE, etc).
Don't put your Groq key in this file — use GROQ_API_KEY or groq_api_key.txt.

Env knobs:
  ROBOT_SERIAL_PORT / ROBOT_TOF_PORT
  ROBOT_LISTEN_SECONDS (default 8)
  ROBOT_SKIP_SERIAL=1  -> voice+LLM only, no hardware
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import serial
import speech_recognition as sr
import whisper
from groq import Groq

# Online neural TTS (Microsoft Sonia and friends) via edge-tts.
try:
    import asyncio
    import edge_tts  # type: ignore
    _EDGE_TTS_AVAILABLE = True
except Exception:
    edge_tts = None  # type: ignore
    _EDGE_TTS_AVAILABLE = False

# Audio playback for the MP3 edge-tts produces.
try:
    from playsound import playsound  # type: ignore
    _PLAYSOUND_AVAILABLE = True
except Exception:
    playsound = None  # type: ignore
    _PLAYSOUND_AVAILABLE = False

# Offline fallback if edge-tts can't reach the network.
try:
    import pyttsx3  # type: ignore
    _PYTTSX3_AVAILABLE = True
except Exception:
    pyttsx3 = None  # type: ignore
    _PYTTSX3_AVAILABLE = False

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BAUD = 115200


def _default_serial_port() -> str:
    if os.name == "nt":
        return "COM3"
    return "/dev/ttyACM0"


PORT = (os.environ.get("ROBOT_SERIAL_PORT") or "").strip() or _default_serial_port()

_WAKE_PHRASE_RAW = os.environ.get("ROBOT_WAKE_PHRASE", "hey mira").strip().lower() or "hey mira"

SYSTEM_PROMPT = """
You are MIRA, a cheerful 6-DOF robotic arm with a witty, friendly personality.
You have a gripper, an overhead camera, a ToF distance sensor, and a sense of humor.
On your table there are three recognizable objects: a pink pig, a green gumby, and a brown otter.

You ALWAYS respond with a single JSON object and NOTHING ELSE (no prose outside JSON, no markdown, no code fences).

REQUIRED FORMAT:
{
  "spoken_response": "Your reply, read aloud by TTS.",
  "robot_action": null
                | {"command": "search_and_pick", "target": "pig"|"gumby"|"otter"}
                | {"command": "point_to", "target": "pig"|"gumby"|"otter"}
                | {"command": "home"}
                | {"command": "wave"}
                | {"command": "show_moves"}
                | {"command": "gesture", "gesture": "nod"|"shake"}
}

VOICE & PERSONALITY:
- Warm, playful, a little cheeky — think a friendly lab assistant who loves their job.
- Use natural, conversational English. Contractions are great. Occasional light humor is welcome.
- For physical commands (grab / home / wave), keep spoken_response to 1-2 lively sentences (e.g. "On it — one pig, coming right up!").
- For general questions / chit-chat / trivia, give a real, helpful answer in 2-5 sentences.
- Never leave spoken_response empty. Never say "I don't know what that is" — if unsure, answer what you can and admit the rest gracefully.

ACTION LOGIC (pick exactly one):
1. User asks to find / grab / pick up / fetch / bring something:
   -> robot_action = {"command": "search_and_pick", "target": ONE_OF[pig, gumby, otter]}.
   -> If they name something else (e.g. "the green guy", "piggy", "the sea animal"), pick the closest match.
2. User asks you to point at / show / indicate / locate / identify where an object is
   ("point to the pig", "where's the otter", "show me the gumby", "can you find the pig for me"):
   -> robot_action = {"command": "point_to", "target": ONE_OF[pig, gumby, otter]}.
   -> IMPORTANT: only use "point_to" when the user clearly just wants you to *indicate* the
      object without picking it up. If they say "grab" / "pick up" / "bring" / "get me" /
      "hand me" — that's rule #1 (search_and_pick), NOT point_to.
3. User asks to sleep / stop / rest / go home / stand down:
   -> robot_action = {"command": "home"}.
4. User praises / thanks / congratulates you ("well done", "good job", "nice work", "thank you", "thanks", "good robot", "good girl", "good boy", "awesome", "perfect", "love you", etc.):
   -> robot_action = {"command": "wave"}, and make spoken_response an upbeat, slightly proud reply.
5. User asks you to show off your movement / dance / wiggle / do a little demo / "show us what you can do" / "show us your moves" / "show us your movement" / "do a dance" / "show off":
   -> robot_action = {"command": "show_moves"}, and make spoken_response a fun, showy intro (e.g. "Oh, you want the full routine? Watch this!").
6. User asks you a clear yes/no question ("is the pig pink?", "do you like waving?", "are you a robot?", "can you see me?", "is the sky blue?", "do you have feelings?"):
   -> Answer the question in spoken_response AND set robot_action = {"command": "gesture",
      "gesture": "nod"} for a YES answer, or {"command": "gesture", "gesture": "shake"} for a NO answer.
   -> Keep spoken_response short and conversational (e.g. "Yep, that's right!" or "Nope, not this time.").
   -> Do NOT use gesture for open-ended / informational questions — only for questions with a
      clean yes-or-no answer.
7. ANY OTHER input — general questions, conversation, jokes, math, science, trivia, advice, small talk, "what time is it", "tell me a fun fact", "who are you", etc.:
   -> robot_action = null, and answer the user's question or engage conversationally like a helpful chatbot. Show some personality.

Never invent new commands. If a request needs hardware you don't have (e.g. "walk over here"), set robot_action to null and explain playfully what you can actually do.
"""

WHISPER_MODEL_NAME = (os.environ.get("ROBOT_WHISPER_MODEL") or "").strip() or "base.en"
print(f"Loading Whisper AI (ears): {WHISPER_MODEL_NAME}")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device="cpu")
recognizer = sr.Recognizer()

_GROQ_KEY_CANDIDATES = (
    re.compile(r'Groq\s*\(\s*api_key\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'api_key\s*=\s*["\'](gsk_[^"\']+)["\']', re.IGNORECASE),
    re.compile(r"(gsk_[A-Za-z0-9]{20,})"),
)


def resolve_groq_api_key() -> str:
    env = os.environ.get("GROQ_API_KEY", "").strip()
    if env:
        return env
    paths: list[Path] = []
    key_file = os.environ.get("GROQ_API_KEY_FILE", "").strip()
    if key_file:
        paths.append(Path(key_file).expanduser())
    here = Path(__file__).resolve().parent
    paths.extend(
        [
            here / "groq_api_key.txt",
            here / "LLM.txt",
            Path.home() / "Downloads" / "LLM.txt",
        ]
    )
    for p in paths:
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        one = text.strip()
        if one.startswith("gsk_"):
            return one.split()[0].strip()
        for pat in _GROQ_KEY_CANDIDATES:
            m = pat.search(text)
            if m:
                return m.group(1).strip()
    raise RuntimeError(
        "Groq API key not found. Set GROQ_API_KEY, or add groq_api_key.txt / LLM.txt "
        "(see docstring at top of robot_llm_voice.py)."
    )


def _wake_phrase_words() -> list[str]:
    return [w for w in re.split(r"\s+", _WAKE_PHRASE_RAW) if w]

def _normalize_wake_variants(text: str) -> str:
    """
    Whisper often mishears "mira" as "mirror"/"meera"/"maera"/etc.
    Normalize common variants so wake detection is reliable.
    """
    if not text:
        return ""
    t = text
    # word-boundary replacements only
    t = re.sub(r"\bmirror\b", "mira", t, flags=re.IGNORECASE)
    t = re.sub(r"\bmirra\b",  "mira", t, flags=re.IGNORECASE)
    t = re.sub(r"\bmeera\b",  "mira", t, flags=re.IGNORECASE)
    t = re.sub(r"\bmaera\b",  "mira", t, flags=re.IGNORECASE)
    t = re.sub(r"\bmeara\b",  "mira", t, flags=re.IGNORECASE)
    # Whisper also commonly mishears "mira" as "miro"/"myro"/"meero".
    t = re.sub(r"\bmiro\b",   "mira", t, flags=re.IGNORECASE)
    t = re.sub(r"\bmyro\b",   "mira", t, flags=re.IGNORECASE)
    t = re.sub(r"\bmeero\b",  "mira", t, flags=re.IGNORECASE)
    t = re.sub(r"\bmira\b",   "mira", t, flags=re.IGNORECASE)
    return t


def _text_contains_wake(text: str) -> bool:
    """Match ROBOT_WAKE_PHRASE using word boundaries (Whisper-friendly)."""
    if not text or not text.strip():
        return False
    text = _normalize_wake_variants(text)
    t = re.sub(r"[^\w\s]", " ", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    words = _wake_phrase_words()
    if not words:
        return False
    if len(words) == 1:
        return bool(re.search(rf"\b{re.escape(words[0])}\b", t))
    pattern = r"\b" + r"\b.*\b".join(re.escape(w) for w in words) + r"\b"
    return bool(re.search(pattern, t))


def _command_after_wake(transcript: str) -> str:
    """
    Return user intent around the wake phrase.
    Prefer words AFTER the wake phrase; if wake appears at the end (common with Whisper),
    fall back to words BEFORE the wake phrase.
    """
    words = _wake_phrase_words()
    if not words:
        return transcript.strip()
    transcript = _normalize_wake_variants(transcript)
    if len(words) == 1:
        m = re.search(rf"\b{re.escape(words[0])}\b", transcript, re.IGNORECASE)
    else:
        pattern = r"\b" + r"\b.*\b".join(re.escape(w) for w in words) + r"\b"
        m = re.search(pattern, transcript, re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    after = transcript[m.end() :].strip()
    after = after.lstrip(",.;:-–— ")
    if after.strip():
        return after.strip()
    before = transcript[: m.start()].strip()
    before = before.rstrip(",.;:-–— ")
    return before.strip()


def _sounddevice_available() -> bool:
    try:
        import sounddevice as sd  # noqa: F401
    except ImportError:
        return False
    return True


def _listen_sounddevice(seconds: float, *, prompt: str) -> str:
    """
    Mic capture without PyAudio using sounddevice.
    Records mono @ 16kHz and transcribes with Whisper (no ffmpeg dependency).
    """
    import sounddevice as sd

    sr_native = 16000
    print(prompt)
    audio_i16 = sd.rec(int(seconds * sr_native), samplerate=sr_native, channels=1, dtype=np.int16)
    sd.wait()

    try:
        print("Processing speech...")
        # Feed raw PCM to Whisper directly (avoids ffmpeg subprocess).
        audio_f32 = (audio_i16.astype(np.float32) / 32768.0).reshape(-1)
        # Simple trimming of leading/trailing near-silence (helps accuracy).
        abs_a = np.abs(audio_f32)
        thr = max(0.01, float(np.quantile(abs_a, 0.75)) * 0.15)
        idx = np.where(abs_a > thr)[0]
        if idx.size > 0:
            a0 = max(int(idx[0]) - int(0.15 * sr_native), 0)
            a1 = min(int(idx[-1]) + int(0.15 * sr_native), audio_f32.shape[0])
            audio_f32 = audio_f32[a0:a1]
        # Stronger decoding settings for accuracy on CPU.
        result = whisper_model.transcribe(
            audio_f32,
            language="en",
            task="transcribe",
            temperature=0.0,
            beam_size=5,
            best_of=5,
            condition_on_previous_text=False,
            fp16=False,
        )
        transcript = (result.get("text") or "").strip()
        print(f"Heard: {transcript}")
        return transcript
    finally:
        pass


def listen_for_wake_phrase_and_command() -> str:
    """One recording: say the wake phrase and your request in the same clip. Returns text after the wake phrase."""
    sec = float(os.environ.get("ROBOT_LISTEN_SECONDS", "8"))
    prompt = f'\nListening... ({sec:.0f}s) Say "{_WAKE_PHRASE_RAW}" and your request in one sentence.'

    if _sounddevice_available():
        transcript = _listen_sounddevice(sec, prompt=prompt)
    else:
        # SpeechRecognition Microphone path requires PyAudio on Windows; keep as a fallback for systems that have it.
        try:
            print(prompt)
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, phrase_time_limit=sec)

            print("Processing speech...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio.get_wav_data())
                tmp_filename = f.name
            try:
                result = whisper_model.transcribe(tmp_filename)
                transcript = (result.get("text") or "").strip()
                print(f"Heard: {transcript}")
            finally:
                try:
                    os.remove(tmp_filename)
                except Exception:
                    pass
        except Exception:
            print("\nNo working mic backend (install sounddevice), falling back to typing.")
            try:
                transcript = input("You: ").strip()
            except EOFError:
                transcript = ""

    if not transcript:
        return ""
    transcript = _normalize_wake_variants(transcript)
    if not _text_contains_wake(transcript):
        print(
            f'No "{_WAKE_PHRASE_RAW}" in that clip — say the wake phrase and your request together.',
            flush=True,
        )
        return ""
    cmd = _command_after_wake(transcript)
    if not cmd.strip():
        print(
            "I heard the wake phrase but nothing after it — add your request in the same sentence.",
            flush=True,
        )
        return ""
    return cmd


TTS_VOICE = os.environ.get("ROBOT_TTS_VOICE", "en-GB-SoniaNeural").strip()
TTS_RATE = os.environ.get("ROBOT_TTS_RATE", "+0%").strip()  # e.g. "+10%" / "-15%"
TTS_VOLUME = os.environ.get("ROBOT_TTS_VOLUME", "+0%").strip()


async def _edge_tts_to_file(text: str, out_path: str) -> None:
    communicate = edge_tts.Communicate(
        text=text,
        voice=TTS_VOICE,
        rate=TTS_RATE,
        volume=TTS_VOLUME,
    )
    await communicate.save(out_path)


def _speak_edge(text: str) -> bool:
    """Synthesize via edge-tts (Sonia) and play. Returns True on success."""
    if not (_EDGE_TTS_AVAILABLE and _PLAYSOUND_AVAILABLE):
        return False
    tmp_path = ""
    try:
        # Each call gets a fresh temp file so playsound never re-grabs a stale handle.
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        asyncio.run(_edge_tts_to_file(text, tmp_path))
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 200:
            return False
        # playsound is blocking on Windows by default — exactly what we want.
        playsound(tmp_path)
        return True
    except Exception as e:
        logging.warning("edge-tts/playsound failed: %s", e)
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _speak_pyttsx3_fallback(text: str) -> bool:
    """Offline fallback. Re-init each call to dodge the Windows pyttsx3 'silent after first call' bug."""
    if not _PYTTSX3_AVAILABLE:
        return False
    try:
        eng = pyttsx3.init()
        try:
            eng.setProperty("rate", 180)
            eng.setProperty("volume", 1.0)
        except Exception:
            pass
        eng.say(text)
        eng.runAndWait()
        try:
            eng.stop()
        except Exception:
            pass
        del eng
        return True
    except Exception as e:
        logging.warning("pyttsx3 fallback failed: %s", e)
        return False


def speak(text: str) -> None:
    """Print + speak the text aloud. Tries edge-tts (Sonia), falls back to pyttsx3, then print-only."""
    if not text:
        return
    print(f"\nRobot: {text}")
    if os.environ.get("ROBOT_DISABLE_TTS", "").strip() in ("1", "true", "yes"):
        return
    if _speak_edge(text):
        return
    if _speak_pyttsx3_fallback(text):
        return
    # If both failed, the print above is the user-visible feedback.


def compute_checksum(s):
    cs = 0
    for c in s:
        cs ^= ord(c)
    return format(cs, "02X")


def build_packet(command, data=""):
    payload = f"{command}|{data}"
    cs = compute_checksum(payload)
    return f"<{payload}|{cs}>\n"


class RobotController:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=1)
        logging.info("Waiting for Arduino to boot...")
        self.wait_for_response(expected="ACK_BOOT", timeout=3.0)

    def send(self, command, data="", wait_for_ack=True):
        packet = build_packet(command, data)
        logging.info(f"TX: {packet.strip()}")
        self.ser.write(packet.encode())
        if wait_for_ack:
            return self.wait_for_response(expected=f"ACK_{command}")
        return True

    def read(self):
        if self.ser.in_waiting:
            line = self.ser.readline().decode().strip()
            logging.info(f"RX: {line}")
            return line
        return None

    def wait_for_response(self, expected="ACK", timeout=2.0):
        start = time.time()
        while time.time() - start < timeout:
            response = self.read()
            if response:
                if expected in response:
                    return True
                elif "ERR" in response:
                    logging.error(f"Arduino error: {response}")
                    return False
            time.sleep(0.01)
        logging.warning(f"Timeout waiting for {expected}.")
        return False


class _DummyRobot:
    def send(self, command, data="", wait_for_ack=True):
        logging.info("SKIP_ROBOT_SERIAL: would send %s | %s", command, data)
        return True

    def read(self):
        return None

    def wait_for_response(self, expected="ACK", timeout=2.0):
        return True


_groq_instance = None


def _groq_client() -> Groq:
    global _groq_instance
    if _groq_instance is not None:
        return _groq_instance
    key = resolve_groq_api_key()
    _groq_instance = Groq(api_key=key)
    return _groq_instance


def chat_with_robot(user_input: str):
    print("Asking Groq (Llama 3.3 70B)...")
    groq_client = _groq_client()
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.75,
            max_tokens=320,
            response_format={"type": "json_object"},
        )
        response_content = chat_completion.choices[0].message.content
        return json.loads(response_content)
    except Exception as e:
        logging.error(f"Groq API Error: {e}")
        return None


def run_demo_sequence(
    *,
    arduino_port: str,
    arduino: "serial.Serial | None" = None,
    overall_timeout: float = 30.0,
) -> bool:
    """
    Fire the Arduino "DEMO" (show-off wiggle) gesture and wait until the
    firmware reports DEMO_COMPLETE / DEMO_ABORTED (or we time out).

    No vision is needed — the Arduino choreographs the whole wiggle itself.
    If a persistent `arduino` connection is passed, we use it (so the board
    is NOT reset on Windows). Otherwise we open a short-lived connection.
    """
    owns_arduino = False
    ser: "serial.Serial | None" = arduino
    try:
        if ser is None or not ser.is_open:
            ser = serial.Serial(arduino_port, BAUD, timeout=0.1)
            owns_arduino = True
            time.sleep(0.2)
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        ser.write(b"DEMO\n")
        ser.flush()

        # Wait for the Arduino to signal completion. DEMO_START is optional;
        # we only block on the terminal status.
        deadline = time.time() + float(overall_timeout)
        buf = ""
        completed = False
        aborted = False
        while time.time() < deadline:
            try:
                chunk = ser.read(64)
            except Exception:
                chunk = b""
            if chunk:
                try:
                    buf += chunk.decode("utf-8", errors="ignore")
                except Exception:
                    pass
                if "DEMO_COMPLETE" in buf:
                    completed = True
                    break
                if "DEMO_ABORTED" in buf:
                    aborted = True
                    break
            else:
                time.sleep(0.05)

        if not (completed or aborted):
            logging.warning("DEMO did not report completion within %.1fs.", overall_timeout)
        return completed
    except Exception as e:
        logging.error("run_demo_sequence failed: %s", e)
        return False
    finally:
        if owns_arduino and ser is not None:
            try:
                ser.close()
            except Exception:
                pass


def run_gesture(
    gesture: str,
    *,
    arduino_port: str,
    arduino: "serial.Serial | None" = None,
    overall_timeout: float = 6.0,
) -> bool:
    """
    Fire a short head-gesture on the Arduino: "nod" (yes) or "shake" (no).
    These are self-contained — no vision, no ToF, just a servo oscillation
    and a return-to-center. Uses the persistent `arduino` connection if
    provided so the board isn't reset on Windows.
    """
    g = (gesture or "").strip().lower()
    mapping = {"nod": "NOD", "shake": "SHAKE"}
    if g not in mapping:
        logging.warning("run_gesture: unknown gesture %r (expected 'nod' or 'shake').", gesture)
        return False
    wire_cmd = mapping[g]
    complete_token = f"{wire_cmd}_COMPLETE"
    aborted_token = f"{wire_cmd}_ABORTED"

    owns_arduino = False
    ser: "serial.Serial | None" = arduino
    try:
        if ser is None or not ser.is_open:
            ser = serial.Serial(arduino_port, BAUD, timeout=0.1)
            owns_arduino = True
            time.sleep(0.15)
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        ser.write(f"{wire_cmd}\n".encode())
        ser.flush()

        deadline = time.time() + float(overall_timeout)
        buf = ""
        completed = False
        aborted = False
        while time.time() < deadline:
            try:
                chunk = ser.read(64)
            except Exception:
                chunk = b""
            if chunk:
                try:
                    buf += chunk.decode("utf-8", errors="ignore")
                except Exception:
                    pass
                if complete_token in buf:
                    completed = True
                    break
                if aborted_token in buf:
                    aborted = True
                    break
            else:
                time.sleep(0.03)

        if not (completed or aborted):
            logging.warning("%s did not report completion within %.1fs.", wire_cmd, overall_timeout)
        return completed
    except Exception as e:
        logging.error("run_gesture(%s) failed: %s", g, e)
        return False
    finally:
        if owns_arduino and ser is not None:
            try:
                ser.close()
            except Exception:
                pass


def run_wave_no_vision(
    *,
    arduino_port: str,
    arduino: "serial.Serial | None" = None,
    overall_timeout: float = 14.0,
) -> bool:
    """
    Fire the Arduino "WAVE" gesture directly (no camera/face search).
    This is intended for the end-of-demo goodbye where we want a quick,
    reliable home->wave->home motion.
    """
    owns_arduino = False
    ser: "serial.Serial | None" = arduino
    try:
        if ser is None or not ser.is_open:
            ser = serial.Serial(arduino_port, BAUD, timeout=0.1)
            owns_arduino = True
            time.sleep(0.2)
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        # Nudge any running behavior to stop before waving.
        try:
            ser.write(b"ABORT\n")
            ser.flush()
            time.sleep(0.15)
        except Exception:
            pass

        ser.write(b"WAVE\n")
        ser.flush()

        deadline = time.time() + float(overall_timeout)
        buf = ""
        completed = False
        aborted = False
        while time.time() < deadline:
            try:
                chunk = ser.read(64)
            except Exception:
                chunk = b""
            if chunk:
                try:
                    buf += chunk.decode("utf-8", errors="ignore")
                except Exception:
                    pass
                if "WAVE_COMPLETE" in buf:
                    completed = True
                    break
                if "WAVE_ABORTED" in buf:
                    aborted = True
                    break
            else:
                time.sleep(0.04)

        if not (completed or aborted):
            logging.warning("WAVE did not report completion within %.1fs.", overall_timeout)
        return completed
    except Exception as e:
        logging.error("run_wave_no_vision failed: %s", e)
        return False
    finally:
        if owns_arduino and ser is not None:
            try:
                ser.close()
            except Exception:
                pass


def open_persistent_arduino(port: str, *, boot_timeout: float = 10.0) -> serial.Serial:
    """
    Open the Arduino serial ONCE and keep it open for the lifetime of the script.

    Every time a new serial.Serial connection is opened on Windows, the Arduino
    typically gets a DTR pulse and resets -> its setup() blocks for ~7s (5s delay
    + smooth-home). That is the primary cause of the huge per-command delay.
    Keeping one connection open means the board only boots once, at startup.
    """
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = BAUD
    ser.timeout = 0.1
    try:
        ser.dtr = False
        ser.rts = False
    except Exception:
        pass
    ser.open()

    # Wait for the firmware's startup banner so we know setup() has finished.
    deadline = time.time() + float(boot_timeout)
    buf = ""
    saw_ready = False
    while time.time() < deadline:
        try:
            chunk = ser.read(128)
        except Exception:
            chunk = b""
        if chunk:
            try:
                buf += chunk.decode("utf-8", errors="ignore")
            except Exception:
                pass
            # FW_BUILD is the last line printed at the end of setup().
            if "FW_BUILD" in buf:
                saw_ready = True
                break
            if "READY" in buf and time.time() - (deadline - boot_timeout) > 4.5:
                # Didn't see FW_BUILD but "READY" arrived and we've waited past the
                # 5s delay(5000) — good enough to proceed.
                saw_ready = True
                break
        else:
            time.sleep(0.05)

    # Print anything useful from the boot banner to the console.
    for line in buf.splitlines():
        line = line.strip()
        if line and ("FW_BUILD" in line or "READY" in line or "Moving to Home" in line):
            print(f"[arduino] {line}")

    if not saw_ready:
        logging.warning(
            "Arduino boot banner not received within %.1fs — proceeding anyway. "
            "If the arm ignores commands, unplug/replug or re-flash the firmware.",
            boot_timeout,
        )
    try:
        ser.reset_input_buffer()
    except Exception:
        pass
    return ser


if __name__ == "__main__":
    try:
        print("\n[SYSTEM STARTUP]")

        # Serial command protocol for the arm is handled by the vision loop (TRACK / GET_COORDS / GRAB / CLAMP / ABORT).
        # We keep ROBOT_SKIP_SERIAL here to allow testing voice+LLM without hardware.
        skip_hw = os.environ.get("ROBOT_SKIP_SERIAL", "").strip().lower() in ("1", "true", "yes")
        if skip_hw:
            logging.info("ROBOT_SKIP_SERIAL set — hardware actions disabled.")

        # Open the Arduino serial connection ONCE. This lets us avoid re-triggering
        # the 7s Arduino boot on every voice command.
        arduino_ser: "serial.Serial | None" = None
        if not skip_hw:
            try:
                print(f"Opening Arduino on {PORT} (waiting for firmware boot)...")
                arduino_ser = open_persistent_arduino(PORT)
                print("Arduino link established.")
            except Exception as e:
                logging.error("Could not open Arduino on %s: %s", PORT, e)
                logging.error("Falling back to per-command opens — expect long delays.")
                arduino_ser = None

        print("Warming up LLM...")
        chat_with_robot("wake up")
        print("Warm-up complete.\n")

        speak(
            "All systems online and I am feeling great. "
            "Just say Hey Mira followed by whatever you need, and I will take it from there."
        )

        from vision_hsv_grab import find_face_and_wave, normalize_target_name, track_and_grab

        while True:
            try:
                user_text = listen_for_wake_phrase_and_command()
                if not user_text:
                    continue

                # End-of-demo scripted goodbye (reliable, no LLM, no camera).
                _t = user_text.lower()
                if (
                    ("say bye" in _t or "say goodbye" in _t or re.search(r"\bbye\b", _t))
                    and ("demo" in _t or "presentation" in _t or "judges" in _t or "help" in _t or "thanks" in _t)
                ):
                    speak(
                        "Thank you, judges, for listening. I hope you enjoyed our demo! "
                        "Bye for now!"
                    )
                    if not skip_hw:
                        run_wave_no_vision(arduino_port=PORT, arduino=arduino_ser)
                    continue

                if "quit" in user_text.lower() or "shut down" in user_text.lower():
                    speak("Alright, powering down. It was a pleasure working with you. Catch you later!")
                    break

                ai_response = chat_with_robot(user_text)

                if ai_response:
                    speak(ai_response.get("spoken_response"))

                    action = ai_response.get("robot_action")
                    if action:
                        cmd = action.get("command")
                        target = str(action.get("target", "")).strip()

                        if cmd == "search_and_pick":
                            tkey = normalize_target_name(target) or target.lower()
                            print(f"Action: search_and_pick target={tkey!r}")
                            if skip_hw:
                                speak(f"(dev mode) I would pick up the {tkey}.")
                            else:
                                try:
                                    ok = track_and_grab(tkey, arduino_port=PORT, arduino=arduino_ser, mode="grab")
                                except Exception as e:
                                    logging.error("Vision/grab failed: %s", e)
                                    speak(
                                        "Hmm, something went sideways on that grab. "
                                        "Maybe check the lighting or give the camera a little nudge, and we can try again."
                                    )
                                else:
                                    if ok:
                                        speak("Got it! One successful grab, delivered with flair.")
                                    else:
                                        speak(
                                            "I had to abort that one — didn't feel confident about the grab. "
                                            "Want me to give it another shot?"
                                        )

                        elif cmd == "point_to":
                            tkey = normalize_target_name(target) or target.lower()
                            print(f"Action: point_to target={tkey!r}")
                            if skip_hw:
                                speak(f"(dev mode) I would point at the {tkey}.")
                            else:
                                try:
                                    ok = track_and_grab(tkey, arduino_port=PORT, arduino=arduino_ser, mode="point")
                                except Exception as e:
                                    logging.error("Vision/point failed: %s", e)
                                    speak("My camera got confused — let me try that one again in a sec.")
                                else:
                                    if ok:
                                        speak("There it is!")
                                    else:
                                        speak("Couldn't lock onto it this time. Want me to try again?")

                        elif cmd == "home":
                            print("Action: returning home...")
                            if skip_hw:
                                speak("(dev mode) Going home.")
                            else:
                                try:
                                    if arduino_ser is not None and arduino_ser.is_open:
                                        arduino_ser.write(b"ABORT\n")
                                        arduino_ser.flush()
                                    else:
                                        with serial.Serial(PORT, BAUD, timeout=0.1) as s:
                                            s.write(b"ABORT\n")
                                            s.flush()
                                except Exception:
                                    pass

                        elif cmd == "wave":
                            print("Action: wave at user")
                            if skip_hw:
                                speak("(dev mode) I would wave back.")
                            else:
                                try:
                                    ok = find_face_and_wave(arduino_port=PORT, arduino=arduino_ser)
                                except Exception as e:
                                    logging.error("Face/wave failed: %s", e)
                                    speak("Oh no, my camera is being shy — I couldn't find a face to wave at.")
                                else:
                                    if not ok:
                                        speak(
                                            "Couldn't quite spot your face in time, but the thought was there! "
                                            "Come a little closer and I will try again."
                                        )

                        elif cmd == "show_moves":
                            print("Action: show_moves (full-body demo wiggle)")
                            if skip_hw:
                                speak("(dev mode) I would show off every joint right now.")
                            else:
                                try:
                                    ok = run_demo_sequence(arduino_port=PORT, arduino=arduino_ser)
                                except Exception as e:
                                    logging.error("Demo sequence failed: %s", e)
                                    speak("Something cramped up mid-routine — let's try that one again in a sec.")
                                else:
                                    if ok:
                                        speak("Ta-da! That's a little sampler of everything I can do.")
                                    else:
                                        speak("Had to cut the routine short, but you get the idea!")

                        elif cmd == "gesture":
                            gesture_kind = str(action.get("gesture", "")).strip().lower()
                            print(f"Action: gesture kind={gesture_kind!r}")
                            if skip_hw:
                                if gesture_kind == "nod":
                                    print("(dev mode) nod")
                                elif gesture_kind == "shake":
                                    print("(dev mode) shake")
                            else:
                                try:
                                    run_gesture(gesture_kind, arduino_port=PORT, arduino=arduino_ser)
                                except Exception as e:
                                    logging.error("Gesture failed: %s", e)
                                    # Gestures are expressive flourishes, not primary
                                    # tasks — don't apologize verbally if the head-bob
                                    # didn't happen; the spoken answer already played.

                    else:
                        print("Action: None")
                else:
                    speak("Sorry, my brain hiccuped and I couldn't pull a reply together. Mind trying that again?")

                time.sleep(0.1)

            except KeyboardInterrupt:
                print("\nExiting.")
                break
    except Exception:
        logging.exception("Fatal error in robot_llm_voice.py")
        if os.name == "nt":
            try:
                input("\nPress Enter to close...")
            except Exception:
                pass
    finally:
        try:
            if "arduino_ser" in dir() and arduino_ser is not None:
                arduino_ser.close()
        except Exception:
            pass
