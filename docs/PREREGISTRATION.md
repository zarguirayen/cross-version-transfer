# Pré-enregistrement — pilote G2 et hypothèses H1 à H4

**Statut : à valider par Rayen, puis figé par le tag git `prereg-v1` avant toute lecture des résultats réels.**
Rédigé le 7 novembre 2026. Analyse figée dans `experiments/g2_preregistered.py` (même commit).
Aucune statistique de résultat n'a été calculée sur les données réelles à la date de rédaction ; seul un
audit de forme a été fait (`docs/DATA_AUDIT.md`). Le script a été testé à blanc sur des données synthétiques.

## Question

Quand on remplace la version d'un modèle ou d'un agent, combien les résultats de l'ancienne version, tâche par
tâche, réduisent-ils le nombre de tests nécessaires pour estimer la fiabilité de la nouvelle, et dans quelles
conditions ce transfert reste-t-il bénéfique ?

## Données

- Sheng & Lu (arXiv 2608.13607), dépôt `jiashengsally/llm-regression-signals`, commit `ce974d7`.
- Unité d'analyse : la **transition** (paire de versions × benchmark).
- Périmètre principal : 30 transitions (5 paires × 6 benchmarks). Analyse secondaire : Llama-2 → Llama-3 (6).
- Type de changement, fixé avant les données : révision mineure = Llama-3 → 3.1, Mistral v0.2 → v0.3 ;
  génération = Qwen2 → 2.5, Qwen2.5 → 3, Gemma 1.1 → 2.
- Population : toutes les questions du benchmark. Cible : le taux de réussite de la nouvelle version sur
  cette population. Historique : les résultats de l'ancienne version sur toutes ces questions (n_v = N).

## Protocole de rejeu

Pour chaque transition et chaque budget m ∈ {10, 20, 40} (m ≤ N/2) : R = 1 000 rejeux ; à chaque rejeu, m
questions tirées uniformément sans remise servent de probes (résultats des deux versions observés).
Graine : 20261107.

Estimateurs, tous avec les mêmes probes :

| Estimateur | Rôle |
|---|---|
| `reset` (Jeffreys pour la moyenne, Clopper-Pearson pour l'intervalle) | référence |
| `full` (héritage complet) | baseline |
| `power_025`, `power_05` (power prior fixe) | baselines |
| `ds_freq` (double échantillonnage fréquentiste) | baseline |
| `bma` (moyenne de modèles, v0.3.2) | **méthode proposée** |

PPI++, commensurate prior et power prior empirique sont prévus mais non implémentés à ce stade : ils seront
ajoutés comme analyses secondaires et signalés comme tels.

## Porte G2

φ² est calculé sur la population complète de chaque transition (intervalle à 95 % par bootstrap des questions).

- **G2 franchie** si, sur les 30 transitions du périmètre, l'étendue de φ² (max − min) est d'au moins 0,20 et
  la médiane de φ² est comprise entre 0,05 et 0,90.
- Si la médiane dépasse 0,90 : l'héritage suffit en pratique ; pivot vers C4 (sécurité).
- Si la médiane est inférieure à 0,05 : l'historique n'aide pas ; le résultat publié devient ce constat.

## Hypothèses et critères

| | Mesure | Critère de succès |
|---|---|---|
| **H1 — Efficacité** | gain observé = EQM(reset) / EQM(bma) ; gain prédit = loi finie 1/(1 − φ² + φ²·m/N) avec φ² de la population | erreur relative médiane \|observé/prédit − 1\| ≤ 0,20 sur les 30 transitions, à m = 20 (critère principal) ; m = 10 et 40 rapportés |
| H1 (secondaire) | même chose avec φ² estimé sur les probes seules | rapporté, sans critère |
| **H2 — Atlas** | modèle additif paire + benchmark sur atanh(φ) | au moins un effet (paire ou benchmark) significatif, Holm à α = 0,05 |
| H2 (exploratoire) | φ² médian révision mineure contre génération | direction attendue : révision mineure > génération ; rapporté sans test (2 contre 3 paires) |
| **H3 — Décision** | choisir v′ si son estimation ≥ taux de v ; regret = meilleur taux − taux choisi ; m = 20 | regret moyen bma < reset (Wilcoxon apparié unilatéral sur les transitions, p < 0,05) **et** regret bma ≤ 1,05 × reset sur au moins 90 % des transitions |
| **H4 — Sécurité** | v′′ = v′ dont 10 points de réussite sont basculés en échecs, identité inchangée ; alerte si la borne haute à 90 % passe sous le taux établi de v′ ; v′ sert d'historique | puissance bma ≥ 0,80 avec fausses alertes ≤ 0,10, à m = 20 ou 40 ; héritage complet et reset rapportés |

Couverture des intervalles à 90 % : rapportée pour chaque estimateur, sans critère de succès.

## Limites connues avant les résultats

- Ce sont des modèles, pas des agents : seul le pattern routage est couvert par ce pilote.
- La « compétence » est le benchmark ; pas de catégorie plus fine.
- L'intervalle de la méthode traite π comme incertain alors que l'historique couvre toute la population : il est
  probablement un peu prudent.
- Les questions d'un même benchmark ne sont pas indépendantes (gabarits communs) ; le bootstrap par question
  peut sous-estimer l'incertitude de φ.
- Décision H3 simplifiée (moyenne contre taux de v), sans zone de probation.
- H4 utilise une dégradation synthétique uniforme ; une dégradation ciblée sur une compétence serait plus
  difficile à détecter.

## Écarts

Tout changement après le gel est consigné dans `docs/DEVIATIONS.md`, avec sa date et sa raison, et signalé
dans le papier.
