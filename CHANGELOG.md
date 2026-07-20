# Changelog

## 7.0.0

- Replaced repeated legacy clinical warnings with a four-dimension clinical summary.
- Separated document completeness/date observations from clinical compatibility.
- Added conservative status aggregation and evidence confidence values.
- Added safety post-processing so local AI cannot mark glucose, thyroid or vitamin tests as matched from pregnancy alone.
- Made deterministic rules the primary result; local Ollama output is a secondary cross-check.
- Prevented `Re-run rules` from starting a long local-model call.
- Automatically migrates legacy `TEST_INDICATION_CHECK`, `PRESCRIPTION_PRESENT`, and `DATE_COVERAGE` records.
- Grouped duplicate test lines and removed one-expander-per-test UI clutter.
- Updated chatbot to report the evidence-gated result first.
- Added configurable Windows port with default `8502`.
- Added 15 automated tests.
