# Pré-enregistrement v2 — confirmation sur des agents (AppWorld)

**Statut : figé par le tag git `prereg-v2` avant toute statistique sur les données AppWorld.**
Rédigé le 7 novembre 2026. Analyse figée dans `experiments/v2_appworld_preregistered.py` (même commit),
testée à blanc sur des données synthétiques au même format.

## Pourquoi une v2

Le pilote v1 (données Sheng & Lu, modèles) a validé la loi d'efficacité (H1) et l'atlas (H2), mais pas H3 ni H4.
Depuis, trois améliorations ont été mises au point **sans toucher aux données AppWorld** :

1. **Estimateur « graphe »** (`graph_logit`) : l'historique de tout l'écosystème d'agents sur les mêmes tâches,
   et pas seulement celui du prédécesseur, sert de variable auxiliaire (régression logistique bayésienne à
   3 paramètres, priors fixés a priori : niveau N(0, 2²), liens N(0, 1²)). Développé sur un monde synthétique
   puis sur les données Sheng & Lu (développement, déjà vues).
2. **Prior P(lien) = 0,9** pour le BMA, choisi par validation croisée par famille sur Sheng & Lu ; analyse
   secondaire.
3. **Détection séquentielle** : lots de 10 probes, jusqu'à 80 ; alerte si la borne haute à 95 % passe sous le
   taux établi ; feu vert si la borne basse dépasse ce taux moins 0,08. Réglée sur du synthétique uniquement.

**Résultats de développement connus avant le gel** (`results/dev_graph_sheng.csv`, données déjà vues, R = 400,
m = 20) : sur les 30 transitions Sheng & Lu, gain médian face au reset ×1,46 pour le graphe contre ×1,26 pour le
BMA ; graphe meilleur que le BMA sur 93 % des transitions ; pire cas ×1,08 ; couverture 90,2 à 95,3 %. Ces
résultats motivent H5 mais ne la prouvent pas.

## Données

- Sorties d'expériences de l'article AppWorld (dépôt `stonybrooknlp/appworld-leaderboard`, fichiers
  `leaderboard.bundle`), 14 agents = 4 architectures (ReAct, PlanExec, FullCodeRefl, IPFunCall) × modèles
  (GPT-4 Turbo, GPT-4o, Llama-3-70B, DeepSeekCoder). Splits `test_normal` (168 tâches) et `test_challenge`
  (417 tâches) ; toutes les tâches communes à tous les agents (vérifié). Résultat : `success` par tâche.
- Seul un audit de forme a été fait (noms d'agents, nombre de tâches, alignement des identifiants). Aucun taux de
  réussite ni aucune corrélation n'a été calculé.

## Transitions (fixées a priori, 20 par split, 40 au total)

| Type | Transitions |
|---|---|
| A — montée de version du modèle | GPT-4 Turbo → GPT-4o, pour chacune des 4 architectures |
| B — changement d'architecture | ReAct → {PlanExec, FullCodeRefl, IPFunCall} avec GPT-4 Turbo et avec GPT-4o ; ReAct → {PlanExec, FullCodeRefl} avec Llama-3 et avec DeepSeekCoder |
| C — changement de famille de modèle | {Llama-3, DeepSeekCoder} → GPT-4o, pour ReAct, PlanExec et FullCodeRefl |

Écosystème de la nouvelle version v′ : tous les autres agents du split, **sauf** v′, son prédécesseur v (qui entre
séparément) et les agents qui utilisent le même modèle que v′ (analyse principale, prudente). Analyse secondaire :
tous les agents sauf v et v′.

## Protocole

R = 400 rejeux par transition et par budget m ∈ {20 (principal), 40}. Probes tirées uniformément sans remise.
Cible : taux de réussite de v′ sur toutes les tâches du split. Estimateurs, mêmes probes : `reset`, `bma`
(P(lien) = 0,5), `bma09` (P(lien) = 0,9), `graph` (écosystème prudent), `graph_all` (écosystème complet).
Incertitude de φ : bootstrap par scénario (3 tâches par scénario). Graine 20261111.

## Hypothèses confirmatoires et critères

| | Mesure | Critère |
|---|---|---|
| **H1-v2 — La loi vaut pour des agents** | erreur relative \|gain observé du BMA / loi finie − 1\| | médiane ≤ 0,20 sur les 40 transitions, m = 20 |
| **H5 — Le graphe fait mieux que le prédécesseur seul** | gain du graphe face au reset, comparé au gain du BMA | Wilcoxon apparié unilatéral p < 0,05 **et** gain du graphe ≥ 0,95 sur ≥ 90 % des transitions **et** couverture moyenne du graphe ≥ 0,88 |
| **H2-v2 — φ² dépend du type de changement** | φ² par transition, types A, B, C | Kruskal-Wallis p < 0,05 ; direction attendue (non testée) : C plus bas |
| **H4-v2 — Détection séquentielle** | agents avec taux ≥ 0,20 ; dégradation synthétique de 10 points ; bruit de ré-exécution de 5 % (bascules symétriques) | puissance ≥ 0,80 **et** probes moyennes ≤ 60 sous dégradation **et** fausses alertes ≤ 0,10 sous bruit |

Rapportés sans critère : couvertures de tous les estimateurs, m = 40, `bma09`, `graph_all`, atlas complet.

## Limites connues avant les résultats

- Beaucoup d'agents ont des taux de réussite très bas (Llama-3, DeepSeekCoder), surtout en `test_challenge` :
  test sévère pour la méthode, dont une limite connue concerne les taux inférieurs à 10 %.
- Toutes les sorties viennent d'une même campagne : pas d'ordre temporel entre agents.
- Les tâches d'un même scénario sont corrélées ; les rejeux tirent des tâches, pas des scénarios.
- H4-v2 repose sur des dégradations et un bruit synthétiques appliqués à des résultats réels.
