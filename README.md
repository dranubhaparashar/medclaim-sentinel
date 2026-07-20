# MedClaim Sentinel v7 — Corrected Clinical Review

MedClaim Sentinel reads medical-claim images using a local Ollama vision model and then performs a conservative, evidence-gated clinical review.

## What v7 corrects

The old repeated output such as `TEST_INDICATION_CHECK — consistent`, `PRESCRIPTION_PRESENT — consistent`, and `DATE_COVERAGE — information` has been replaced by four clear dimensions:

1. Diagnosis / treatment context
2. Diagnosis ↔ prescribed medicine
3. Prescribed medicine ↔ billed medicine
4. Diagnosis ↔ investigation

Matched and informational results are no longer displayed as warning expanders. Tests are grouped in one table, repeated service lines are counted, confidence is always shown, and dates/document-presence observations are kept separate from clinical matching.

The final decision is produced by curated deterministic rules using evidence extracted by the local vision model. The Ollama clinical response is shown only as a secondary cross-check and cannot override missing evidence.

## Correct expected result for the included sample

- Diagnosis/context: **MATCHED** — pregnancy / antenatal care is documented.
- Diagnosis ↔ prescribed medicine: **INSUFFICIENT_EVIDENCE** — no reliable itemized prescription medicine list is visible.
- Prescription ↔ billed medicine: **INSUFFICIENT_EVIDENCE** — the pharmacy schedule has bill numbers, dates and amounts, but not medicine names.
- Diagnosis ↔ investigations: **PARTIALLY_MATCHED**.
  - Pregnancy-specific ultrasound/FHR/NT/dual-marker/anomaly tests: matched.
  - CBC and urine routine: matched in antenatal context.
  - FBS, PPBS, HbA1c, thyroid profile, vitamin B12 and vitamin D: conditional unless the corresponding diagnosis, abnormal result, risk factor or prescriber note is documented.

The product does not approve or reject a claim automatically.

## Upgrade an existing project

Stop Streamlit, copy the v7 patch contents into the existing project, and replace matching files. Keep the `storage` folder.

```powershell
cd "C:\Users\Anubha\Documents\projects\medclaim-sentinel"
.\run_windows.ps1
```

The default port is now `8502` so it can run beside another Streamlit application on `8501`.

Use another port when needed:

```powershell
.\run_windows.ps1 -Port 8503
```

Then open:

```text
http://localhost:8502
```

Open **Claim 360 → Clinical checks**. Existing legacy checks are upgraded automatically. Press **Refresh clinical rules** once if the old labels remain. Use **Read remaining pages with local AI** only when documents still need local extraction.

## Local AI behavior

- The local vision model extracts text and structured evidence from the original images.
- Curated clinical rules generate the primary clinical result.
- The local clinical model is a secondary cross-check.
- A bill amount or bill number is never converted into a medicine name.
- Pregnancy alone does not prove diabetes, hypothyroidism, vitamin B12 deficiency or vitamin D deficiency.
- Missing itemized medicine evidence produces `INSUFFICIENT_EVIDENCE`, not `MATCHED`.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The v7 package includes 15 automated tests and does not require Ollama for testing.
