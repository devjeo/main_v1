"""
cane_boot.py -- the ONE entry point (use this on boot / in systemd).

    python cane_boot.py [any ai_stream.py options, e.g. --conf 0.6]

Self-contained: replaces cane_setup.py. Flow:
  1. List saved Wi-Fi profiles and wait for NetworkManager to connect.
  2. No CANE_DEVICE_SECRET in cane.env (not paired)?
       -> speak a prompt, scan the camera for the app's QR, join the
          hotspot in it, check internet, claim this device with the
          single-use token (Supabase links the account's user id to this
          cane's device id), save the secret to cane.env.
  3. Check internet (informational only).
  4. Replace this process with ai_stream.py (camera is free by then).

QR (plain JSON from the phone app):
    {"v": 1, "ssid": "MyHotspot", "pw": "hotspot-password", "tok": "<token>"}
Supabase RPC expected:
    claim_device_with_token(p_device_id, p_token) -> {"device_secret": "..."}
A paired cane with no Wi-Fi still starts: detection + speech work offline
and cane_cloud's heartbeat reconnects by itself later.
Needs: nmcli (Raspberry Pi OS Bookworm), cane.env with SUPABASE_URL and
SUPABASE_ANON_KEY.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, Optional

import requests

logger = logging.getLogger("cane_boot")

_HERE = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(_HERE, "cane.env")
AI_STREAM = os.path.join(_HERE, "ai_stream.py")
PIPER_DIR = "./piper"
TTS_CACHE_DIR = "./tts_cache"

BT_SPEAKER_MAC = "F4:75:EA:25:74:8E"   # AB4085 (demo speaker); "" to disable. Env/cane.env CANE_SPEAKER_MAC overrides.
BT_CONNECT_TIMEOUT_S = 15

WIFI_WAIT_S = 25           # wait for a saved network at boot
REMIND_EVERY_S = 60
SCAN_INTERVAL_S = 0.4
WIFI_JOIN_TIMEOUT_S = 40
WIFI_JOIN_ATTEMPTS = 3
INTERNET_WAIT_S = 20

PHRASE_NEEDS_SETUP = "Setup needed. Open the Smart Cane app and show the setup code to the camera."
PHRASE_RECEIVED = "Code received. Connecting to your hotspot."
PHRASE_BAD_QR = "That is not a Smart Cane setup code."
PHRASE_NO_WIFI = "I could not connect to your hotspot. Turn it on, then show the code again."
PHRASE_NO_NET = "Connected, but there is no internet. Check your hotspot, then show a new code."
PHRASE_FAIL = "Pairing failed. Please show a new code from the app."
PHRASE_DONE = "Pairing complete. Your cane is ready."
SETUP_PHRASES = [PHRASE_NEEDS_SETUP, PHRASE_RECEIVED, PHRASE_BAD_QR,
                 PHRASE_NO_WIFI, PHRASE_NO_NET, PHRASE_FAIL, PHRASE_DONE]


# ----------------------------------------------------------------------
# cane.env read / write
# ----------------------------------------------------------------------
def read_env(path: str = ENV_FILE) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                values[key.strip()] = val.strip().strip('"').strip("'")
    for key in ("SUPABASE_URL", "SUPABASE_ANON_KEY", "CANE_DEVICE_ID", "CANE_DEVICE_SECRET"):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def update_env_file(updates: Dict[str, str], path: str = ENV_FILE) -> None:
    """Sets/adds keys, keeps other lines/comments. Atomic, owner-only (holds a secret)."""
    lines = []
    if os.path.exists(path):
        with open(path) as f:
            lines = f.read().splitlines()
    out, done = [], set()
    for line in lines:
        s = line.strip()
        key = s.split("=", 1)[0].strip() if "=" in s and not s.startswith("#") else None
        if key in updates:
            out.append(f"{key}={updates[key]}")
            done.add(key)
        else:
            out.append(line)
    for key, val in updates.items():
        if key not in done:
            out.append(f"{key}={val}")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(out) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def get_device_id(env: Dict[str, str]) -> Optional[str]:
    """cane.env value if set, else derived from the Pi's CPU serial / machine-id."""
    if env.get("CANE_DEVICE_ID"):
        return env["CANE_DEVICE_ID"]
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("Serial"):
                    serial = line.split(":", 1)[1].strip()
                    if serial and set(serial) != {"0"}:
                        return "CANE-" + serial[-8:].upper()
    except OSError:
        pass
    try:
        with open("/etc/machine-id") as f:
            return "CANE-" + f.read().strip()[:8].upper()
    except OSError:
        return None


# ----------------------------------------------------------------------
# QR: camera input is untrusted, validate hard
# ----------------------------------------------------------------------
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_\-]{16,128}$")
_qr_detector = None


def _has_control_chars(s: str) -> bool:
    return any(ord(c) < 32 or ord(c) == 127 for c in s)


def parse_setup_qr(text: str) -> Optional[Dict[str, str]]:
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or type(data.get("v")) is not int or data["v"] != 1:
        return None
    ssid, pw, tok = data.get("ssid"), data.get("pw"), data.get("tok")
    if not all(isinstance(x, str) for x in (ssid, pw, tok)):
        return None
    if not 1 <= len(ssid.encode("utf-8")) <= 32 or _has_control_chars(ssid):
        return None
    if not 8 <= len(pw) <= 63 or _has_control_chars(pw):
        return None
    if not _TOKEN_RE.match(tok):
        return None
    return {"ssid": ssid, "pw": pw, "tok": tok}


def decode_qr(frame) -> Optional[str]:
    global _qr_detector
    import cv2
    if _qr_detector is None:
        _qr_detector = cv2.QRCodeDetector()
    try:
        text, _pts, _straight = _qr_detector.detectAndDecode(frame)
    except cv2.error:
        return None
    return text or None


# ----------------------------------------------------------------------
# Wi-Fi (nmcli). Args always as a list, never via a shell, so a hostile
# SSID/password in a QR can't inject commands. Password is never logged.
# ----------------------------------------------------------------------
def _run(cmd, timeout: float = 20):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return subprocess.CompletedProcess(cmd, 1, "", str(e))


def wifi_connected() -> bool:
    r = _run(["nmcli", "-t", "-f", "TYPE,STATE", "device"], timeout=5)
    for line in r.stdout.splitlines():
        kind, _, state = line.partition(":")
        if kind == "wifi" and state.startswith("connected"):
            return True
    return False


def saved_wifi_profiles():
    r = _run(["nmcli", "-t", "-f", "NAME,TYPE", "connection", "show"], timeout=5)
    out = []
    for line in r.stdout.splitlines():
        name, _, kind = line.rpartition(":")
        if kind == "802-11-wireless":
            out.append(name)
    return out


def wait_for_wifi(seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if wifi_connected():
            return True
        time.sleep(1)
    return False


def join_wifi(ssid: str, password: str):
    """Joins and saves (autoconnect) the network. Returns (ok, message)."""
    _run(["nmcli", "connection", "delete", "id", ssid])  # so a changed password takes effect
    message = ""
    # wpa-psk covers WPA2 and WPA2/WPA3 mixed; sae is WPA3-only.
    # Hidden is tried last because it makes nmcli probe for the SSID directly.
    variants = [("wpa-psk", False), ("sae", False), ("wpa-psk", True)]
    for attempt in range(1, WIFI_JOIN_ATTEMPTS + 1):
        _run(["nmcli", "device", "wifi", "rescan"])
        time.sleep(3)
        key_mgmt, hidden = variants[(attempt - 1) % len(variants)]
        _run(["nmcli", "connection", "delete", "id", ssid])
        add = ["nmcli", "connection", "add", "type", "wifi",
               "con-name", ssid, "ssid", ssid,
               "connection.autoconnect", "yes",
               "wifi-sec.key-mgmt", key_mgmt,
               "wifi-sec.psk", password]
        if hidden:
            add += ["802-11-wireless.hidden", "yes"]
        r = _run(add)
        if r.returncode == 0:
            r = _run(["nmcli", "--wait", str(WIFI_JOIN_TIMEOUT_S), "connection", "up", "id", ssid],
                     timeout=WIFI_JOIN_TIMEOUT_S + 15)
        message = (r.stderr or r.stdout).strip()
        if r.returncode == 0:
            return True, message
        logger.warning("Wi-Fi join attempt %d/%d (%s%s) failed: %s", attempt, WIFI_JOIN_ATTEMPTS,
                       key_mgmt, ", hidden" if hidden else "", message)
        _run(["nmcli", "connection", "delete", "id", ssid])  # don't leave a broken profile saved
        time.sleep(4)
    return False, message


def wait_for_internet(base_url: str, seconds: float = INTERNET_WAIT_S) -> bool:
    """Any HTTP answer from Supabase (even 401) means the network path works."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            requests.get(base_url, timeout=4)
            return True
        except requests.RequestException:
            time.sleep(2)
    return False


class ClaimError(Exception):
    pass


def claim_device(base_url: str, anon_key: str, device_id: str, token: str) -> str:
    """Trades the one-time token for this cane's secret."""
    try:
        r = requests.post(
            f"{base_url}/rest/v1/rpc/claim_device_with_token",
            headers={"apikey": anon_key, "Authorization": f"Bearer {anon_key}",
                     "Content-Type": "application/json"},
            json={"p_device_id": device_id, "p_token": token},
            timeout=10,
        )
    except requests.RequestException as e:
        raise ClaimError(f"request failed: {e}") from e
    if not r.ok:
        raise ClaimError(f"HTTP {r.status_code}: {r.text[:200]}")
    try:
        data = r.json()
    except ValueError as e:
        raise ClaimError("server returned non-JSON") from e
    secret = data.get("device_secret") if isinstance(data, dict) else data
    if not isinstance(secret, str) or not secret:
        raise ClaimError("server did not return a device secret")
    return secret


# ----------------------------------------------------------------------
# pairing
# ----------------------------------------------------------------------
def try_pair(payload: Dict[str, str], say) -> bool:
    env = read_env()
    base_url = (env.get("SUPABASE_URL") or "").rstrip("/")
    anon_key = env.get("SUPABASE_ANON_KEY")
    device_id = get_device_id(env)
    if not (base_url and anon_key and device_id):
        logger.error("Pairing needs SUPABASE_URL and SUPABASE_ANON_KEY in cane.env")
        say(PHRASE_FAIL)
        return False

    say(PHRASE_RECEIVED)
    ok, message = join_wifi(payload["ssid"], payload["pw"])
    if not ok:
        logger.warning("Could not join hotspot %r: %s", payload["ssid"], message)
        say(PHRASE_NO_WIFI)
        return False
    if not wait_for_internet(base_url):
        say(PHRASE_NO_NET)
        return False

    secret = None
    for attempt in range(1, 4):
        try:
            secret = claim_device(base_url, anon_key, device_id, payload["tok"])
            break
        except ClaimError as e:
            logger.warning("Claim attempt %d/3 failed: %s", attempt, e)
            if "HTTP 4" in str(e):   # server said no (bad/expired/used token): retrying won't help
                break
            time.sleep(2)
    if secret is None:
        say(PHRASE_FAIL)
        return False

    update_env_file({"CANE_DEVICE_ID": device_id, "CANE_DEVICE_SECRET": secret})
    logger.info("Paired as %s", device_id)
    say(PHRASE_DONE)
    return True


def pair_with_qr(say) -> bool:
    """Scans the camera until a valid QR pairs the cane. Ctrl+C to quit."""
    import cv2
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("Could not open the camera (/dev/video0).")
        return False

    last_remind = last_scan = last_bad = float("-inf")
    last_text, ignore_until = None, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            now = time.monotonic()
            if now - last_remind >= REMIND_EVERY_S:
                last_remind = now
                say(PHRASE_NEEDS_SETUP)
            if now - last_scan < SCAN_INTERVAL_S:
                continue
            last_scan = now

            text = decode_qr(frame)
            if not text or (text == last_text and now < ignore_until):
                continue   # nothing, or the same code still held up
            last_text, ignore_until = text, now + 60

            payload = parse_setup_qr(text)
            if payload is None:
                if now - last_bad > 10:
                    last_bad = now
                    say(PHRASE_BAD_QR)
                continue
            if try_pair(payload, say):
                return True
            last_remind = time.monotonic()   # don't talk over the failure message
    finally:
        cap.release()


def connect_speaker() -> None:
    """Best effort: ask the Pi to connect to the paired Bluetooth speaker.
    Never blocks boot for long and never raises -- the cane just speaks
    from the Pi's own output if the speaker isn't available."""
    mac = (read_env().get("CANE_SPEAKER_MAC") or os.environ.get("CANE_SPEAKER_MAC") or BT_SPEAKER_MAC).strip()
    if not mac or shutil.which("bluetoothctl") is None:
        return
    if not re.fullmatch(r"([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", mac):
        print(f"[boot] ignoring invalid speaker address {mac!r}")
        return
    _run(["bluetoothctl", "power", "on"], timeout=5)
    _run(["bluetoothctl", "trust", mac], timeout=5)
    r = _run(["bluetoothctl", "connect", mac], timeout=BT_CONNECT_TIMEOUT_S)
    ok = "Connection successful" in (r.stdout or "") or "already connected" in (r.stdout or "").lower()
    print(f"[boot] Bluetooth speaker {mac}: {'connected' if ok else 'not connected (using the Pi output)'}")
    if ok:
        time.sleep(2.0)   # give the audio system a moment to switch its output


def wait_until_spoken(audio, timeout: float = 10.0) -> None:
    end = time.monotonic() + timeout
    time.sleep(0.4)
    while time.monotonic() < end:
        q = getattr(audio, "_queue", None)
        if q is not None and q.empty() and getattr(audio, "_current_priority", -1) == -1:
            return
        time.sleep(0.2)


# ----------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    have_nmcli = shutil.which("nmcli") is not None

    connect_speaker()   # so the setup prompts and the stream speak through it

    if have_nmcli:
        profiles = saved_wifi_profiles()
        print(f"[boot] saved Wi-Fi profiles: {profiles or 'none'}")
        if profiles:
            print("[boot] waiting for Wi-Fi...")
            print("[boot] connected" if wait_for_wifi(WIFI_WAIT_S) else "[boot] no saved network in range")
    else:
        print("[boot] nmcli not found -- skipping Wi-Fi/pairing (dev machine?)")

    if have_nmcli and not read_env().get("CANE_DEVICE_SECRET"):
        print("[boot] cane not paired -> starting QR setup")
        audio = None
        say = lambda text: print(f"[setup] {text}")
        try:
            from pi_audio import AudioOutput, Priority
            audio = AudioOutput(piper_dir=PIPER_DIR, cache_dir=TTS_CACHE_DIR)
            audio.start()
            audio.precache(SETUP_PHRASES)
            say = lambda text: (print(f"[setup] {text}"), audio.speak(text, Priority.INFO))
        except Exception as e:   # no voice -> print-only is fine for setup
            print(f"[warn] No speech ({e}); showing messages on screen only.")
        try:
            if not pair_with_qr(say):
                sys.exit(1)
            if audio is not None:
                wait_until_spoken(audio)
        except KeyboardInterrupt:
            print("\nStopped.")
            sys.exit(130)
        finally:
            if audio is not None:
                audio.stop()
        time.sleep(1.0)   # let the camera device release

    url = (read_env().get("SUPABASE_URL") or "").rstrip("/")
    if url:
        online = wait_for_internet(url, seconds=5)
        print(f"[boot] internet: {'OK' if online else 'offline (will keep working locally)'}")

    print("[boot] starting ai_stream.py")
    os.execv(sys.executable, [sys.executable, AI_STREAM, *sys.argv[1:]])


if __name__ == "__main__":
    main()