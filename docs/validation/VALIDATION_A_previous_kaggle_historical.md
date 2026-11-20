# Validation record A: previous Kaggle environment (HISTORICAL)

Environment: the first Kaggle session (host 8fc9019f4a9e), 2x Tesla T4, driver 580.178.04, CUDA 13.0, glibc 2.39.
Dates: 2026-11-20 (several runs the same day). These results are history and are **not** evidence about any other server.

| Item | Result |
|---|---|
| Full automated suite on the GPU host (MAPLE_LIVE=1) | 56 tests, OK, 120.5 s (41 earlier tests + 15 HTTP-API tests) |
| Clean restore into a separate directory (from local archives) | 29/29 probe checks (manifests/clean_restore_report.json) |
| Restore with simulated Dataset mount (deep path) | restore steps all ok, 4 specialists + quant answered, clean shutdown |
| Local recovery verifier (PC) | 22/22 |
| Specialists validated by real load | deepseek, gemma, qwen3, fast |
| Validated context (tested, per model) | fast 32768, qwen3 32768, gemma 131072, deepseek 65536; deepseek 131072 NOT validated |

Not covered by A: any other machine, Dataset-based restoration, VS Code / Continue / Open WebUI.
