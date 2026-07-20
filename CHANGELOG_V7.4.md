# MedClaim Sentinel v7.4

- Added permanent claim deletion for accidental, test, incomplete and half-loaded entries.
- Added an incomplete-entry filter in Dashboard → Delete or clean up claim entries.
- Added a protected Delete this claim panel in Claim 360.
- Deletion requires the exact claim number, acknowledgement and a reason.
- Removes uploaded files, OCR data, clinical checks, duplicate links, correction proposals and active audit events.
- Keeps a minimal governance deletion log and recalculates claims that were linked as duplicates.
- Local AI is never started by the delete action.
