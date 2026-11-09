# Résultats v8 — patterns agentiques, systèmes multi-agents et évaluation asynchrone

Exécutée le 9 novembre 2026 après le gel `prereg-v8` (empreintes vérifiées après exécution). Script figé :
`experiments/v8_patterns_async_preregistered.py`, R = 200, graine 20261123. Sorties : `results/v8/`.
13 agents AppWorld publiés après l'article (jamais analysés), 2 splits, 26 évaluations ; Multi-SWE-bench (analyse
secondaire). 9 hypothèses : 6 soutenues, 3 non soutenues.

## Verdict, selon les critères figés

| | Critère | Résultat | Verdict |
|---|---|---|---|
| **N1 — La méthode bat le hasard (nouveaux agents)** | médiane > 1, Wilcoxon p < 0,05 | ×1,31 (min ×0,91), 24/26 ; p = 3,7e−7 | soutenue |
| **N2 — Y compris les systèmes multi-agents** | idem, 8 évaluations | ×1,18 (min ×0,91) ; p = 0,008 | soutenue |
| **N3 — Calibrée** | couverture ≥ 0,85 | 0,959 (min 0,915) ; multi-agents 0,949 | soutenue |
| **N4 — Loi du plan stratifié** | erreur médiane ≤ 0,20 | 0,081 | soutenue |
| **A1 — Le biais d'échéance existe** | médiane > 0, p < 0,05 | +6,3 points (moyenne +7,2) ; p = 3,8e−6 | soutenue |
| **A2 — Le graphe le corrige** | Wilcoxon p < 0,05 et rapport RMSE > 1 | biais absolu 0,063 → 0,084 ; p = 0,74 ; rapport 0,86 | **non soutenue** |
| **A3 — Calibré à l'échéance** | couverture ≥ 0,85 | 0,770 (min 0,27) | **non soutenue** |
| **P1 — Même pattern, mêmes échecs** | Wilcoxon sur 7 langages | +0,044 ; 6/7 ; p = 0,023 | soutenue |
| **P2 — Changer de pattern efface plus d'information** | Wilcoxon sur 7 langages | φ² +0,037 ; 5/7 ; p = 0,11 | **non soutenue** |

Rapportés sans critère : gains par classe (multi-agents ×1,18 ; RL ×1,22 ; agents simples ×1,41) ; décisions
« le nouvel agent bat-il le meilleur agent antérieur ? » sur 20 évaluations à écart ≥ 5 points : erreur 4,8 % pour la
méthode, 3,0 % pour le tirage uniforme, 2,1 % pour la moyenne stratifiée ; k = 15 : biais naïf +5,0 points, graphe
−2,7 points, couverture 0,866. Deux pertes : LARA (challenge) ×0,91, SynthAPI (challenge) ×0,94.

## Diagnostic exploratoire (après le gel, `experiments/v8_exploratory_async.py`)

- Les agents de l'article échouent plus souvent là où ils sont lents (effet de lenteur γ ≈ −0,67 en logit, transmis
  par le graphe) ; les nouveaux agents, non (γ propre médian +0,02). L'écart prédit le biais de la correction
  (Spearman −0,83). Le biais naïf restant vient de ce que les tâches lentes sont les plus difficiles.
- Variante exploratoire, non confirmée : imputer les probes manquantes par la difficulté seule (γ = 0) réduit l'erreur
  du naïf d'un facteur 1,23 (médiane), biais −3,2 points, couverture 0,86 (min 0,40).

## Ce que cela autorise à écrire

La méthode (le graphe choisit, GREG croisé estime) tient sur des agents de patterns différents, y compris des systèmes
multi-agents traités comme un seul nœud, avec des intervalles calibrés et une loi qui prédit le gain. Deux agents de
même pattern échouent davantage ensemble. Une évaluation arrêtée à une échéance est biaisée vers le haut ; la
correction apprise sur d'anciens agents ne se transfère pas aux nouveaux : c'est un problème ouvert.
