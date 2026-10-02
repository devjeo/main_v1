"""
cane_cloud.py - device-side link between the Smart Cane (Raspberry Pi) and
Supabase. Right now it does one job: keep the device's online status honest.

  * Every HEARTBEAT_INTERVAL_S seconds it calls the `device_heartbeat` RPC.
    The SERVER stamps last_seen_at (the Pi has no battery-backed clock).
  * On a normal shutdown (Ctrl+C, `systemctl stop`, SIGTERM) it stops the
    heartbeat and calls `device_set_offline` so the app flips immediately.
  * On a crash / power loss / Wi-Fi drop it can't say goodbye - the
    server-side sweeper (pg_cron) marks it offline after ~45s of silence.

Security: this uses the public ANON key plus a per-device secret. It never
needs the service-role key or any Firebase key. The secret only works with
the two RPC functions created by supabase_online_status.sql.

Config comes from environment variables, or a `cane.env` file next to this
script (KEY=VALUE per line; real env vars win):

    SUPABASE_URL=https://xxxx.supabase.co
    SUPABASE_ANON_KEY=...
    CANE_DEVICE_ID=...
    CANE_DEVICE_SECRET=...        (from: select provision_device_secret('ID'))
"""
import atexit
import os
import signal
import sys
import threading

import requests

HEARTBEAT_INTERVAL_S = 15
REQUEST_TIMEOUT_S = 5

_HERE = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(_HERE, "cane.env")


def _load_env_file(path):
    values = {}
    if not os.path.exists(path):
        return values
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def _config(name, file_values):
    return os.environ.get(name) or file_values.get(name)


class CaneCloud:
    def __init__(self, battery_reader=None):
        """battery_reader: optional zero-arg callable returning an int 0-100
        (or None if unknown). Called on every heartbeat."""
        file_values = _load_env_file(ENV_FILE)
        self.url = (_config("SUPABASE_URL", file_values) or "").rstrip("/")
        self.anon_key = _config("SUPABASE_ANON_KEY", file_values)
        self.device_id = _config("CANE_DEVICE_ID", file_values)
        self.secret = _config("CANE_DEVICE_SECRET", file_values)
        self.battery_reader = battery_reader

        self._session = requests.Session()
        self._stop = threading.Event()
        self._thread = None
        self._shutdown_done = False
        self._failing = False  # only print on state changes, not every beat

    # ------------------------------------------------------------------
    @property
    def configured(self):
        return all([self.url, self.anon_key, self.device_id, self.secret])

    def _rpc(self, name, payload):
        return self._session.post(
            f"{self.url}/rest/v1/rpc/{name}",
            headers={
                "apikey": self.anon_key,
                "Authorization": f"Bearer {self.anon_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=REQUEST_TIMEOUT_S,
        )

    # ------------------------------------------------------------------
    def _read_battery(self):
        if self.battery_reader is None:
            return None
        try:
            level = self.battery_reader()
            return None if level is None else max(0, min(100, int(level)))
        except Exception as e:
            print(f"[cloud] battery read failed: {e}")
            return None

    def _beat(self):
        try:
            r = self._rpc("device_heartbeat", {
                "p_device_id": self.device_id,
                "p_secret": self.secret,
                "p_battery": self._read_battery(),
            })
            if r.ok:
                if self._failing:
                    print("[cloud] heartbeat recovered - device online")
                self._failing = False
            else:
                if not self._failing:
                    print(f"[cloud] heartbeat rejected: HTTP {r.status_code}: {r.text[:200]}")
                self._failing = True
        except requests.RequestException as e:
            if not self._failing:
                print(f"[cloud] heartbeat failed (will keep retrying): {e}")
            self._failing = True

    def _loop(self):
        while not self._stop.is_set():
            self._beat()
            self._stop.wait(HEARTBEAT_INTERVAL_S)

    # ------------------------------------------------------------------
    def start(self):
        """Start the heartbeat thread. Returns False (and does nothing) if
        the cloud config is missing, so the stream still works offline."""
        if not self.configured:
            print("[cloud] not configured (need SUPABASE_URL, SUPABASE_ANON_KEY, "
                  "CANE_DEVICE_ID, CANE_DEVICE_SECRET) - heartbeat disabled")
            return False
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="cane-heartbeat", daemon=True)
        self._thread.start()
        print(f"[cloud] heartbeat started for {self.device_id} "
              f"(every {HEARTBEAT_INTERVAL_S}s)")
        return True

    def shutdown(self):
        """Stop heartbeating FIRST (so it can't flip us back online), then
        tell the server we're going offline. Safe to call more than once."""
        if self._shutdown_done or not self.configured:
            return
        self._shutdown_done = True
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=REQUEST_TIMEOUT_S + 1)
        try:
            r = self._rpc("device_set_offline", {
                "p_device_id": self.device_id,
                "p_secret": self.secret,
            })
            if r.ok:
                print("[cloud] marked offline")
            else:
                print(f"[cloud] could not mark offline: HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            # Fine - the server-side sweeper will catch it within ~45s.
            print(f"[cloud] could not mark offline ({e}); sweeper will handle it")

    def install_shutdown_hooks(self):
        """Run shutdown() on normal exit, Ctrl+C, and SIGTERM (systemctl stop)."""
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))