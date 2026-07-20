MedClaim Sentinel v7.2 UI fix

Replace your existing project app.py with this app.py.

Fixes:
- Removes repeated raw clinical check expanders.
- Hides raw JSON from the normal reviewer view.
- Aggregates duplicate test findings.
- Shows four clear clinical summary cards.
- Restores clickable clinical reference links.
- Hides stale LOCAL_AI_UNAVAILABLE from the main clinical result.
- Keeps date/document observations in a collapsed informational section.
- Supports old and new process_claim() signatures.
- Removes the unnecessary "no paid API" status/marketing text.
