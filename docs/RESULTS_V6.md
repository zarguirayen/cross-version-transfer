# Résultats v6 — confirmation de la méthode unifiée (LiveCodeBench, TheAgentCompany)

Exécutée le 7 novembre 2026 après le gel `prereg-v6` (empreintes vérifiées). Script figé :
`experiments/v6_unified_preregistered.py`, R = 400, graine 20261119. 29 transitions jamais analysées :
27 versions successives de modèles sur LiveCodeBench (713 problèmes), 2 versions d'agents OpenHands sur
TheAgentCompany (175 tâches de bureau). Sorties : `results/v6/`.

## Verdict, selon les critères figés (m = 20)

| | Critère | Résultat | Verdict |
|---|---|---|---|
| **U1 — Estimer : robuste et calibré** | pire gain ≥ 0,80 ; couverture ≥ 0,88, min ≥ 0,80 | **pire gain ×1,45** ; médiane ×2,20 ; couverture 0,944 (min 0,895) | soutenue |
| **U2 — Estimer : mieux que le prédécesseur seul** | Wilcoxon p < 0,05 | ×2,20 contre ×1,56 ; p = 2,3 × 10⁻⁶ | soutenue |
| **U3 — Décider : mieux que le tirage au hasard** | Wilcoxon p < 0,05 | erreur 6,0 % contre 8,8 % (17 transitions) ; p = 0,016 | soutenue |
| **U4 — Les deux formes de la loi** | erreurs médianes ≤ 0,20 | φ² : 0,064 ; plan stratifié : 0,072 | soutenue |

## Ce que montrent les données

- **La méthode unifiée tient sur deux sources nouvelles et deux natures de nœuds** (modèles et agents de bureau).
  Avec 20 tâches choisies par le graphe, l'estimation GREG croisée vaut 44 tâches tirées au hasard, ne perd sur
  aucune transition (pire cas ×1,45) et ses intervalles à 90 % couvrent la vraie valeur 94 % du temps.
- **TheAgentCompany**, agents qui naviguent, écrivent et traitent des documents : ×1,46 sur les deux transitions,
  couverture 0,96. Deux transitions seulement : une illustration, pas une preuve.
- **La loi** est confirmée pour la 6e fois (φ²) et la 3e fois (plan stratifié).

## Réserves honnêtes

- **Aucune version sous 10 % de réussite** dans ces données : la robustesse dans ce régime reste établie par la v5
  seulement.
- **Le décideur choisi n'est pas le meilleur ici.** Erreurs de décision (écart ≥ 5 points) : reset uniforme 8,8 %,
  graphe stratifié (choisi) 6,0 %, GREG croisé 4,3 %, simple moyenne stratifiée 4,1 %. En v5, c'était l'inverse
  (GREG mauvais décideur pour les versions quasi nulles). Ce qui améliore les décisions dans les deux cas, c'est le
  **choix des tâches par le graphe** ; le choix de l'estimateur pour décider n'est pas tranché.
- Les dates de sortie des modèles viennent de la mémoire de l'assistant ; LiveCodeBench évalue des modèles.

## Bilan des six confirmations

| Étape | Données | Transitions | Erreur de la loi φ² | Meilleur gain médian |
|---|---|---|---|---|
| v1 (pilote) | modèles, Sheng & Lu | 30 | 2,6 % | ×1,26 |
| v2 | agents AppWorld | 40 | 2,7 % | ×1,40 |
| v3 | SWE-bench Verified | 89 | 3,6 % | ×2,01 |
| v4 | SWE-bench Lite | 28 | 3,9 % | ×2,14 |
| v5 | 9 jeux, 7 langages | 33 | 5,8 % | ×1,61 |
| v6 | LiveCodeBench, TheAgentCompany | 29 | 6,4 % | ×2,20 |
