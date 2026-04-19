# ED Boarding Study Pipeline

This folder contains a local pipeline for:

`Risk-stratified harm of emergency department boarding among initially non-high-acuity admissions`

Expected inputs on this machine:

- `C:\Users\adamk\Desktop\mimic-iv-3.1\mimic-iv-3.1\mimic-iv-3.1`
- `C:\Users\adamk\Desktop\Charls\NHANES鏁版嵁\analysis_ascii\input\mimic_iv_ed_2_2`
- `C:\Users\adamk\Desktop\mimic-iv-3.1\MC-MED`

Main script:

- `run_ed_boarding_study.py`

Implemented outputs:

- MIMIC base cohort for adult ED admissions with triage acuity 3-5
- Exclusion of likely direct ED-to-ICU admissions using an adjustable grace window
- 6-hour landmark cohort for target trial emulation
- Shared-core early features from triage, 0-2h vitals, 0-2h labs, and medication reconciliation
- MC-MED transport cohort and aligned shared-core features
- MIMIC risk-model development and internal validation
- MC-MED score transport output
- Cross-fitted doubly robust ATE and ATT estimates for `boarded_6h`

Current encoded assumptions:

- Direct ED-to-ICU is approximated as first ICU admission within 2 hours after ED departure.
- The main predictive and causal outcome is unexpected ICU transfer within 24 hours after ED departure, with a 48-hour secondary endpoint.
- The DML landmark analysis uses only information available in the first 0-2 hours after ED arrival.
- MC-MED is prepared for risk-score transport and process analysis. A definitive external validation of the same ICU-upgrade endpoint is not attempted unless a reliable encounter-level ICU-upgrade label is later derived.
