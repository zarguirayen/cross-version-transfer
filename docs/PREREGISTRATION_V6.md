# Pré-enregistrement v6 — méthode unifiée (LiveCodeBench, TheAgentCompany)

**Statut : figé par le tag git `prereg-v6` avant toute statistique de résultat sur ces données.**
Rédigé le 7 novembre 2026. Analyse figée dans `experiments/v6_unified_preregistered.py` et `experiments/v6_data.py`.
Testée à blanc avec `--fake`.

## Méthode unifiée (issue de la v5, aucun réglage depuis)

1. **Choisir les tâches avec le graphe** : tâches triées par difficulté vue par les nœuds antérieurs, m = 20
   strates, 1 tâche par strate.
2. **Estimer** le taux de la nouvelle version : GREG croisé (`graph_greg_cf`, état du code au commit de ce gel,
   identique à `prereg-v5`).
3. **Décider** (garder v ou passer à v′) : signe de p̂′ − p_v avec le graphe stratifié (`graph_v3` sur les mêmes
   probes stratifiées).

Les rôles sont fixés avant les données : on ne choisit pas, a posteriori, l'estimateur le plus favorable.

## Données (audit de forme seulement)

| Source | Nœuds | Tâches | Transitions | Nature |
|---|---|---|---|---|
| LiveCodeBench (`LiveCodeBench/submissions`, commit 6ca212e) | 68 modèles (7 sans date sûre, exclus des écosystèmes) | 713 problèmes communs | 27, en 12 gammes de produit | versions successives de modèles (GPT-4 → GPT-4o, o1 → o3, Claude 3.5 → 4, Gemini 1.5 → 2.5, DeepSeek, Qwen, Nemotron) |
| TheAgentCompany (`TheAgentCompany/experiments`, commit f9b1e94) | 17 agents | 175 tâches de bureau | 2 (OpenHands 0.14 → 0.28, Gemini et Claude) | agents qui naviguent, communiquent, traitent des documents |

Règles complètes dans `v6_data.py` : résultat LiveCodeBench = premier échantillon correct ; TheAgentCompany =
tâche entièrement réussie ; doublons exclus ; noms de tâches normalisés ; écosystème = nœuds de date strictement
antérieure, sauf v.

**Limites déclarées avant les résultats**
- Les dates de sortie des modèles (table `RELEASE`) ont été écrites de mémoire par l'assistant ; une erreur de
  date ne change que la composition de l'écosystème. Dates incertaines : nœud exclu.
- Les scores agrégés publics de LiveCodeBench ont pu être vus pendant l'entraînement de l'assistant ; aucun
  résultat par problème n'a été ouvert.
- LiveCodeBench évalue des modèles, pas des agents ; TheAgentCompany n'apporte que 2 transitions (rapportées
  séparément, incluses dans les tests).
- Écosystèmes de 3 à 10 nœuds pour les quatre transitions GPT-4.

## Protocole

R = 400 rejeux, m ∈ {20 (principal), 40}. Plans uniforme et stratifié tirés indépendamment. Estimateurs rapportés :
`reset_u`, `bma_u`, `v3_u`, `reset_s`, `v3_s`, `greg_cf_s`. Graine 20261119.

## Hypothèses confirmatoires (m = 20, 29 transitions)

| | Mesure | Critère |
|---|---|---|
| **U1 — Estimer : robuste et calibré** | `greg_cf_s` | pire gain ≥ 0,80 **et** couverture moyenne ≥ 0,88 **et** minimale ≥ 0,80 |
| **U2 — Estimer : mieux que le prédécesseur seul** | gain `greg_cf_s` contre gain `bma_u` | Wilcoxon apparié unilatéral p < 0,05 |
| **U3 — Décider : mieux que le tirage au hasard** | erreur de décision `v3_s` contre `reset_u`, transitions où \|p′ − p_v\| ≥ 0,05 | Wilcoxon apparié unilatéral p < 0,05 |
| **U4 — Les deux formes de la loi** | erreur médiane loi φ² (BMA) et loi du plan stratifié (`reset_s`) | les deux ≤ 0,20 |

Rapportés sans critère : m = 40 ; par source ; par gamme ; largeurs d'intervalles ; versions sous 10 %.
