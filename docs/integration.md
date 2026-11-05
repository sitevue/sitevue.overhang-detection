# Connecting the customer UI

Run the service next to the camera (the machine with the GPU):

```bash
overhang serve rtsp://CAMERA/stream --port 8000 --cors-origin http://localhost:3000
```

For UI development without a camera, replay a recorded video at belt speed, forever:

```bash
overhang serve recordings/fpl-2026-10-29.mp4 --realtime --loop --port 8000 --cors-origin http://localhost:3000
```

## Endpoints

| | | |
|---|---|---|
| `GET` | `/health` | `{status: starting/running/finished/error, error, trays, uptime_s, source}` |
| `GET` | `/api/trays/latest` | Last tray result (or `null`) |
| `GET` | `/api/trays?limit=50` | Recent results, newest first |
| `GET` | `/api/summary?window=100` | Counts and % per class over the last N **reliable** trays, per-side totals, mean meat offset, and the latest tray's class + light |
| `GET` | `/api/config` | Class cut-offs and light colours in use |
| `WS` | `/ws/trays` | One JSON message per tray, pushed when ready |

Tray result fields: [output.md](output.md). The UI only needs `overhang_class`, `light`,
`reliable` (and `tray_id`/`timestamp`); everything else is diagnostics.

## Browser example

```js
const ws = new WebSocket("ws://overhang-box:8000/ws/trays");
ws.onmessage = (e) => {
  const tray = JSON.parse(e.data);
  // tray.overhang_class: "none" | "mild" | "small" | "large" | "unknown"
  // tray.light: "green" | "yellow" | "red" | null
  showTray(tray.tray_id, tray.overhang_class, tray.light, tray.reliable);
};
ws.onclose = () => setTimeout(connect, 2000);   // reconnect; the service keeps running

// line summary every few seconds
setInterval(async () => {
  const s = await (await fetch("http://overhang-box:8000/api/summary?window=100")).json();
  // s.share_pct = {none: 77.0, mild: 18.0, small: 4.0, large: 1.0}
  showSummary(s.share_pct, s.latest?.light);
}, 5000);
```

## Behaviour to know

- A slow or disconnected WebSocket client never slows the pipeline: if a client falls 100
  messages behind, its oldest messages are dropped. Use `/api/trays` to back-fill.
- The service keeps the last 2,000 results in memory only; nothing is written to disk. Persist
  results in the UI's backend if the customer needs history.
- If the pipeline stops (camera lost, crash), `/health` shows `status: error` with the reason.
  Restart the service (e.g. systemd/Docker restart policy).
- Review images are off by default. `--debug-dir DIR` writes one overlay per tray (green outer
  edge, magenta inner edge, orange/yellow meat outside) for checking the system, not for customers.
