# Pré-enregistrement v8 — patterns agentiques, systèmes multi-agents et évaluation asynchrone

**Statut : figé par le tag git `prereg-v8` avant toute statistique de résultat sur les 13 nouveaux agents AppWorld
et avant toute statistique par pattern sur Multi-SWE-bench.**
Rédigé le 9 novembre 2026. Analyse figée dans `experiments/v8_patterns_async_preregistered.py`, données dans
`experiments/v8_data.py`, estimateur asynchrone dans `src/cvt/asynchronous.py` (même commit). Testée à blanc avec
`--fake` (données synthétiques sans lien entre durée et succès ni structure par pattern : A1, P1 et P2 non détectées,
comme attendu).

## Questions

1. **Patterns agentiques et systèmes multi-agents.** La méthode (le graphe choisit les tâches, GREG croisé estime)
   reste-t-elle meilleure que le tirage au hasard et calibrée pour des agents de patterns différents, y compris des
   systèmes multi-agents ?
2. **Évaluation asynchrone.** Quand les résultats des probes arrivent dans le désordre et qu'on s'arrête à une
   échéance, l'estimation sur les probes revenues est-elle biaisée, et le graphe permet-il de corriger ce biais ?
3. **Structure par pattern.** Deux agents de même pattern échouent-ils davantage sur les mêmes tâches que deux agents
   de patterns différents ?

## Données

- **AppWorld, 13 agents soumis au classement après l'article (avril 2025 à octobre 2026), jamais analysés** :
  fichiers `leaderboard.bundle` officiels de `stonybrooknlp/appworld-leaderboard`, déchiffrés avec le paquet
  `appworld` (mot de passe public du paquet), non redistribués. Splits `test_normal` (168 tâches) et
  `test_challenge` (417), soit 26 évaluations. Succès : fichier d'évaluation ; pour IBM-HCL-GP, qui n'en a pas,
  règle d'AppWorld appliquée aux rapports par tâche (au moins un test, aucun échec), vérifiée identique au fichier
  d'évaluation sur 4 agents qui ont les deux. Étapes : nombre d'« Environment Interaction » dans
  `logs/environment_io.md`.
- **Graphe de chaque nouvel agent** : tous les agents publiés strictement avant lui (date du fichier du classement ;
  date de `metadata.json` pour IBM-HCL-GP, absent du fichier), dont les 14 agents de l'article. Prédécesseur quand la
  lignée est explicite : CUGA GPT-4.1 → CUGA GLM-5.3-Flash, LARA gpt-4.1-mini → LARA claude-opus-4-7,
  ReAct GPT-4o → ReAct + 2 SetBSR demos.
- **Classes de pattern**, fixées d'après la description de chaque soumission avant tout résultat
  (`v8_data.NEW_AGENTS`) : *multi-agent* (LARA MAS : un explorateur écrit un plan, des exécutants spécialisés par
  application ; IBM CUGA : planificateur–exécutant hiérarchique, un contrôleur délègue à des sous-agents
  spécialisés), 4 agents, 8 évaluations ; *RL-trained* (LOOP, SynthAPI, AgentRL, RooAGI), 8 évaluations ;
  *single-agent* (AtlasHarness, Button Agent, capybara, ReAct + démos, IBM-HCL-GP), 10 évaluations.
- **Multi-SWE-bench**, 7 langages, plan croisé initial du 20250329 (3 échafaudages × 9 modèles), déjà utilisé en v5
  (estimation) et en v7 (structure par échafaudage et par modèle). **Aucune statistique par pattern n'y a été
  calculée.** Patterns : MSWE-agent et MopenHands sont des boucles d'agent ; MagentLess est un flux fixe de type
  Agentless.

**Audit fait avant le gel** : noms, dates, nombre de tâches, alignement des identifiants, présence des journaux
d'étapes, accord de la règle de lecture des rapports. Aucun taux de réussite ni aucune corrélation n'a été calculé
sur les nouveaux agents.

## Développement (connu avant le gel, données déjà vues)

`experiments/dev_v8_async.py`, les 14 agents de l'article (résultats vus en v2, nombres d'étapes jamais analysés),
R = 60 : à l'échéance k = 10 sur m = 20, l'estimation naïve surestime le taux de +0,100 en moyenne ; l'estimateur
du graphe ramène le biais à +0,001, divise l'erreur quadratique moyenne par 1,95 (médiane) et couvre 0,880 (min 0,783).
À k = 15 : +0,050 contre +0,007, rapport 1,40, couverture 0,890. Effet de lenteur dans le graphe : γ ≈ −1,0 (logit).
Une variante sans incertitude sur la capacité couvrait 0,84 ; elle a été corrigée avant le gel. Gain de GREG croisé
sur ces agents en développement v7 : ×1,26.

## Méthode

Pour chaque nouvel agent et chaque split : R = 200 rejeux, m = 20. Tirage uniforme (reset, prior hybride) ; plan
stratifié sur la difficulté du graphe (une probe par strate) ; GREG croisé. Gain = EQM du tirage uniforme / EQM de
la méthode. **Asynchrone** : sur les mêmes probes stratifiées, ordre d'arrivée = nombre d'étapes de l'agent (égalités
au hasard) ; échéance quand k probes sont revenues (k = 10, principal ; k = 15, rapporté). Estimateur naïf : moyenne
des probes revenues. Estimateur du graphe (`cvt.asynchronous.deadline_estimate`) : modèle de lenteur ajusté sur le
graphe, capacité du nouvel agent estimée sur ses probes revenues, probes non revenues imputées, intervalle avec
incertitudes de plan, d'imputation, de capacité et de γ. Graine 20261123.

## Hypothèses confirmatoires

| | Mesure | Critère |
|---|---|---|
| **N1 — La méthode bat le hasard sur les nouveaux agents** | gain de GREG croisé, 26 évaluations | médiane > 1 et Wilcoxon unilatéral du log-gain > 0, p < 0,05 |
| **N2 — Y compris les systèmes multi-agents** | même mesure, 8 évaluations multi-agents | médiane > 1 et p < 0,05 |
| **N3 — Calibrée** | couverture des intervalles à 90 % | moyenne ≥ 0,85 sur tout et sur les multi-agents |
| **N4 — La loi du plan stratifié tient** | erreur relative du gain prédit (reset stratifié) | médiane ≤ 0,20 |
| **A1 — Le biais d'échéance existe** | biais signé de l'estimation naïve, k = 10 | médiane > 0 et Wilcoxon > 0, p < 0,05 |
| **A2 — Le graphe le corrige** | biais absolu naïf moins biais absolu du graphe ; rapport des RMSE | Wilcoxon > 0, p < 0,05, et rapport médian > 1 |
| **A3 — Calibré à l'échéance** | couverture de l'estimateur du graphe, k = 10 | moyenne ≥ 0,85 |
| **P1 — Même pattern, mêmes échecs** | résidus (modèle à une dimension, comme en v7), paires d'échafaudages et de modèles différents : même pattern moins patterns différents | Wilcoxon unilatéral sur 7 langages, p < 0,05 |
| **P2 — Changer de pattern efface plus d'information** | φ² brut, paires de même modèle : pattern conservé moins pattern changé | Wilcoxon unilatéral sur 7 langages, p < 0,05 |

Rapportés sans critère : résultats par classe de pattern, k = 15, décisions « le nouvel agent bat-il le meilleur
agent antérieur ? », corrélation étapes–succès, largeurs.

## Limites connues avant les résultats

- Un système multi-agents est un seul nœud : le graphe ne voit pas ses agents internes. Seuls deux cadres
  multi-agents (4 agents, 8 évaluations) : N2 est un test de faible puissance.
- Le nombre d'étapes remplace le temps de calcul réel ; pour un système multi-agents il compte les interactions avec
  l'environnement, pas les échanges entre agents.
- « Asynchrone » désigne l'évaluation (résultats qui arrivent dans le désordre), pas l'architecture des agents.
- P1 et P2 sont des analyses secondaires sur des données déjà utilisées pour d'autres questions ; dans ce plan, le
  pattern « flux fixe » n'a qu'un seul échafaudage (MagentLess).
- Les dates de certaines entrées du classement sont des dates de soumission ; l'ordre entre agents publiés le même
  jour n'est pas utilisé.
