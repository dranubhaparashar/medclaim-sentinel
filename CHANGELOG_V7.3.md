# v7.3 Fast UI

- Removed automatic claim-pipeline migration from Claim 360 page navigation.
- Prevented the old v6 `process_claim()` fallback from silently launching Ollama.
- Included a compatible pipeline where local clinical AI runs only when explicitly requested.
- Older clinical findings are rendered immediately with reconstructed four-dimension cards.
- **Refresh clinical rules** runs deterministic rules only.
- **Read remaining pages with local AI** is the only Claim 360 action that invokes Ollama.
- Existing SQLite data and uploaded documents remain unchanged.
