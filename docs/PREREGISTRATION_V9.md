# Pré-enregistrement v9 (prospectif) — l'estimateur d'échéance sur des agents pas encore publiés

**Statut : figé par le tag git `prereg-v9` le 9 novembre 2026, avant qu'aucun des agents analysés n'existe sur le
classement AppWorld.** Analyse figée dans `experiments/v9_prospective_deadline.py` (même commit). Testée à blanc avec
`--fake` (données synthétiques) et sur l'instantané actuel, qui ne contient encore aucun agent éligible.

## Pourquoi

En v8, le biais d'échéance est confirmé (A1), mais la correction qui transférait l'effet de lenteur des anciens agents
a échoué (A2, A3). Le développement après v8, sur les données v8 déjà vues (`experiments/dev_v9_deadline.py`), désigne
une variante plus simple : imputer les probes non revenues par la difficulté seule (γ = 0). Sur ces données : rapport
des RMSE naïf / corrigé 1,30 (médiane), couverture 0,856 (min 0,26), biais −3,3 points. Ces chiffres ne sont pas une
preuve ; seule la confirmation ci-dessous en sera une.

## Données et règle d'exécution

- **Agents de confirmation** : tous les agents du classement AppWorld (`stonybrooknlp/appworld-leaderboard`) dont la
  date est **postérieure au 2026-11-04** (dernier instantané analysé en v8), sur les deux splits, déchiffrés comme en
  v8. Succès, étapes et dates : mêmes règles qu'en v8.
- **Graphe** de chaque agent : tous les agents publiés strictement avant lui (les 14 de l'article, les 13 de v8, et
  les nouveaux plus anciens que lui).
- **Quand exécuter** : une seule fois, sur le premier instantané qui contient au moins 5 agents éligibles, ou sur
  l'instantané du 2027-05-01 s'il vient avant. Avec moins de 5 agents, les résultats sont rapportés comme descriptifs.
  Aucune autre exécution sur ces données avant celle-ci.

## Méthode

Identique à v8 : m = 20, plan stratifié, R = 200, graine 20261127 ; ordre d'arrivée = nombre d'étapes de l'agent ;
échéance k = 10. Estimateur corrigé : `cvt.asynchronous.deadline_estimate` avec γ = 0 (aucun effet de lenteur).

## Hypothèses confirmatoires

| | Mesure | Critère |
|---|---|---|
| **V1 — La méthode bat le hasard** (réplication de N1) | gain de GREG croisé | médiane > 1 et Wilcoxon du log-gain, p < 0,05 |
| **D1 — Le biais d'échéance existe** (réplication de A1) | biais signé naïf, k = 10 | médiane > 0 et Wilcoxon, p < 0,05 |
| **D2 — La correction par la difficulté réduit l'erreur** | RMSE naïf − RMSE corrigé ; rapport | Wilcoxon > 0, p < 0,05, et rapport médian > 1 |
| **D3 — Calibrée à l'échéance** | couverture à 90 % de l'estimateur corrigé | moyenne ≥ 0,85 |

## Limites connues

Le nombre d'agents futurs est inconnu ; avec peu d'agents, la puissance sera faible. Le développement a montré un agent
très mal couvert (0,26) : D3 peut échouer même si D2 réussit.
