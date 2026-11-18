# V1.4 experiment LOCK

config_sha256: 1600c02d147b8da517cfa5d30c836666bf9b718b740b898d597dca42dffb5259
preflight_sha256: bc2ee732f2beffe2182f43dcdada4362951a6c718feb3b3900560c5eba1c1779
reviewed_code_commit: 8ce81289688db4b234241891634c60378fb975d5
common_alpha: 0.174
reviewer_type: AI
human_signatures: not_provided
gpu_return: manual_user
automatic_time_limit_stop: false

V1.4 overrides dormant original boost/budget/32-row fields. Only greedy/C/D/E test runs below are authorized.
D-C is confirmatory; D-E descriptive; E-C point only. No re-selection after rehearsal.
Development check: {"dev_anal_greedy_and_gradients_previously_completed": true, "independent_evaluation": false, "n": 200, "scope": "development_pipeline_check", "separate_CDE_rehearsal_omitted": true, "split": "dev_select"}
When selection data are reused, their performance is selection-biased and is not independent validation.
Original two-person human extraction/LOCK signatures were not provided; AI review is not human signoff.

```json
[
  {
    "condition": "greedy",
    "problem_count": 1319,
    "rows": 1,
    "run_config": "configs/generated/budget120/final/test_greedy_aNA_T0.00_NA_s0_k1_v2_35a0b069422900bb.yaml",
    "run_config_sha256": "03d8ddf3ad8d8c43e1996fc754ef7f349047fa9247626cf32b74677c5013a106",
    "run_id": "test_greedy_aNA_T0.00_NA_s0_k1_v2_35a0b069422900bb"
  },
  {
    "condition": "C",
    "problem_count": 1319,
    "rows": 16,
    "run_config": "configs/generated/budget120/final/test_C_a0.1740_T0.50_B0_s0_k16_v2_bbae766c5128f03e.yaml",
    "run_config_sha256": "688a972b5cfe1c3b53713a3e89a5eb2723f7362da3b735ec27c44b8801e66fe6",
    "run_id": "test_C_a0.1740_T0.50_B0_s0_k16_v2_bbae766c5128f03e"
  },
  {
    "condition": "D",
    "problem_count": 1319,
    "rows": 16,
    "run_config": "configs/generated/budget120/final/test_D_a0.1740_T0.50_B0_s0_k16_v2_bbae766c5128f03e.yaml",
    "run_config_sha256": "8849ce754ceaf9c2f298313eec374ce65be8dad5a7dbe43c11ab28dfdeb69a6a",
    "run_id": "test_D_a0.1740_T0.50_B0_s0_k16_v2_bbae766c5128f03e"
  },
  {
    "condition": "E",
    "problem_count": 1319,
    "rows": 16,
    "run_config": "configs/generated/budget120/final/test_E_a0.1740_T0.50_B0_s0_k16_v2_bbae766c5128f03e.yaml",
    "run_config_sha256": "87e019319571b1f556112ed76cd854a36ffea09cd23d98b0bbe867893a8344a3",
    "run_id": "test_E_a0.1740_T0.50_B0_s0_k16_v2_bbae766c5128f03e"
  }
]
```
