# Décisions du noyau sur la lignée OpenHands (20 tâches testées par version)

| Nouvelle version | Version courante | Estimation (intervalle à 90 %) | Vrai taux | Décision | Sûre | Bonne décision |
|---|---|---|---|---|---|---|
| 20250203_openhands_4x_scaled | 20241029_OpenHands-CodeAct-2.1-sonnet-20241022 | 0.597 [0.500 ; 0.689] | 0.608 | promote | non | oui |
| 20250415_openhands | 20250203_openhands_4x_scaled | 0.596 [0.465 ; 0.716] | 0.658 | keep | non | non |
| 20250520_openhands_devstral_small | 20250415_openhands | 0.545 [0.359 ; 0.721] | 0.468 | keep | non | oui |
| 20250524_openhands_claude_4_sonnet | 20250520_openhands_devstral_small | 0.745 [0.641 ; 0.830] | 0.704 | promote | oui | oui |
| 20250716_openhands_kimi_k2 | 20250524_openhands_claude_4_sonnet | 0.673 [0.559 ; 0.773] | 0.654 | keep | non | oui |
| 20250805_openhands-Qwen3-Coder-30B-A3B-Instruct | 20250716_openhands_kimi_k2 | 0.530 [0.418 ; 0.640] | 0.516 | keep | oui | oui |
| 20250805_openhands-Qwen3-Coder-480B-A35B-Instruct | 20250805_openhands-Qwen3-Coder-30B-A3B-Instruct | 0.652 [0.501 ; 0.783] | 0.696 | promote | non | oui |
| 20250807_openhands_gpt5 | 20250805_openhands-Qwen3-Coder-480B-A35B-Instruct | 0.685 [0.562 ; 0.790] | 0.718 | keep | non | non |
| 20251127_openhands_claude-opus-4-5 | 20250807_openhands_gpt5 | 0.779 [0.614 ; 0.894] | 0.776 | promote | non | oui |

9 versions ; intervalle contenant le vrai taux : 100% ; bonnes décisions : 78% ; tâches testées : 180 au lieu de 4500.
