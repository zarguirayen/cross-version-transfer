# Pré-enregistrement v3 — confirmation sur SWE-bench Verified (versions réelles d'agents, ordre temporel réel)

**Statut : figé par le tag git `prereg-v3` avant toute statistique de résultat sur ces données.**
Rédigé le 7 novembre 2026. Analyse figée dans `experiments/v3_swebench_preregistered.py` et
`experiments/v3_swebench_data.py` (même commit). Testée à blanc avec `--fake` (résultats aléatoires de même forme).

## Pourquoi une v3

La v2 (AppWorld) a confirmé la loi, l'atlas et l'apport du graphe, avec une limite : versions qui réussissent
moins de 10 % des tâches. Depuis, **sur données déjà vues uniquement** (AppWorld après v2, Sheng & Lu après v1),
deux corrections de principe et une garde ont été mises au point (`src/cvt/graph.py`, `graph_v3`) :

1. **Population finie** : la cible est le taux réalisé sur les N tâches ; les probes y entrent par leur résultat
   observé, les autres par un tirage de Bernoulli. L'intervalle inclut l'incertitude de prédiction.
2. **Moyenne de modèles** : les 4 sous-ensembles de covariables (aucune, écosystème, prédécesseur, les deux),
   pondérés par leur vraisemblance marginale (échantillonnage d'importance), prior uniforme sur les modèles.
3. **Garde d'identifiabilité** : si les probes ont toutes le même résultat, aucune pente n'est identifiable ;
   on revient au double échantillonnage sur le seul prédécesseur.

Résultats de développement connus avant le gel (`results/dev_v3_*.csv`, données déjà vues, R = 200, m = 20) :
voir la section « Développement » plus bas. Ils motivent H5-v3 et H6-v3 mais ne les prouvent pas. Une variante
à prior empirique de pente (tiré des autres agents) a été essayée sur AppWorld puis abandonnée, faute de gain net.

## Données

- Dépôt public `SWE-bench/experiments`, dossier `evaluation/verified` (clone du 7 novembre 2026) : 175 soumissions
  avec résultats par tâche, 500 tâches. Résultat = tâche résolue (`resolved`), 0 sinon.
- Audit de forme uniquement : univers de 500 tâches commun, aucun identifiant résolu hors univers, noms, dates et
  métadonnées. Aucun taux, aucune corrélation calculés.
- Limite connue : des taux globaux de certains agents figurent dans le classement public et ont pu être vus pendant
  l'entraînement de l'assistant ; les résultats par tâche et leurs corrélations, jamais.

## Graphe, lignées et transitions (règles mécaniques, `v3_swebench_data.py`)

- Nœud = une soumission, datée par son préfixe AAAAMMJJ. Lignée = même produit (table `PRODUCTS`, écrite à partir
  des noms de dossiers) ; pour mini-SWE-agent, même fournisseur de modèle.
- Transition v → v′ : v = la soumission la plus récente de la même lignée à une date strictement antérieure.
  Résultat : **89 transitions** (60 de produit, 29 mini-SWE-agent), liste dans le script.
- Écosystème de v′ : toutes les soumissions de date strictement antérieure à v′, sauf v (aucune information du
  futur). Analyse secondaire : sans la lignée de v′.

## Protocole

R = 400 rejeux par transition et par budget m ∈ {20 (principal), 40}. Probes uniformes sans remise.
Estimateurs, mêmes probes : `reset`, `bma` (prédécesseur seul, confirmé en v2), `graph` (v2, figé),
`eco_only`, `graph_v3`, `graph_v3_strict`. φ : bootstrap par dépôt GitHub. Graine 20261114.

## Hypothèses confirmatoires et critères (m = 20, 89 transitions)

| | Mesure | Critère |
|---|---|---|
| **H1-v3 — La loi tient** | \|gain du BMA / loi finie − 1\| | médiane ≤ 0,20 |
| **H5-v3 — Le graphe v3 fait mieux que le prédécesseur seul** | gain de `graph_v3` face au reset vs gain du BMA | Wilcoxon apparié unilatéral p < 0,05 **et** gain ≥ 0,95 sur ≥ 90 % des transitions **et** couverture moyenne ≥ 0,88 |
| **H6-v3 — Pas de perte catastrophique** | pire transition | gain de `graph_v3` ≥ 0,80 **et** couverture ≥ 0,80 sur toutes les transitions |
| **H7-v3 — Le prédécesseur ajoute à l'écosystème** | gain de `graph_v3` contre `eco_only` (régression logistique sur la seule difficulté vue par l'écosystème, de type IRT, même population finie, garde → reset) | Wilcoxon apparié unilatéral p < 0,05 |
| **H2-v3 — L'arête de lignée porte une information propre** | φ²(v, v′) contre φ²(c, v′), c = agent antérieur hors lignée, de taux le plus proche de v | Wilcoxon apparié unilatéral p < 0,05 |
| **H4-v3 — Détection de régressions réelles** | transitions où v′ perd ≥ 10 points ; même règle séquentielle qu'en v2 | testée seulement si ≥ 5 telles transitions : alerte ≥ 0,80 **et** fausses alertes ≤ 0,10 sur les transitions sans baisse |

Rapportés sans critère : m = 40, `graph` v2, `graph_v3_strict`, taux de repli de la garde, par type (produit /
mini), atlas complet, probes équivalentes (gain × m).

## Développement (connu avant le gel, données déjà vues)

R = 200, m = 20. Gain médian face au reset (pire cas) ; couverture moyenne (minimum).

| Données | BMA | graphe v2 | graphe v3 avec garde |
|---|---|---|---|
| AppWorld, 40 transitions | ×1,11 (×0,77) ; 0,937 (0,875) | ×1,38 (×0,37) ; 0,914 (0,690) | ×1,32 (×0,67) ; 0,926 (0,865) |
| Sheng & Lu, 30 transitions | ×1,26 (×1,02) ; 0,928 (0,875) | ×1,40 (×0,93) ; 0,909 (0,835) | ×1,37 (×0,95) ; 0,916 (0,835) |

Le v3 échange environ 4 % de gain médian contre un pire cas bien meilleur et une couverture plus régulière.
Variantes écartées en développement : repli dès moins de 2 succès (coûte ×1,37 → ×1,27 sur AppWorld) ; prior
empirique de pente (pas de gain net).

**Constat de développement qui motive H7-v3** (`experiments/dev_h7_appworld.py`, AppWorld) : avec l'écosystème
complet, l'arête de lignée n'apporte rien de mesurable au-delà de l'écosystème seul (×1,30 contre ×1,26,
Wilcoxon p = 0,38). H7-v3 teste donc, sur des données neuves et en ordre temporel réel, si le prédécesseur ajoute
une information propre au graphe. Le résultat attendu d'après le développement est un échec.

## Limites connues avant les résultats

- Les tâches d'un même dépôt sont corrélées ; les rejeux tirent des tâches.
- Les lignées sont des regroupements par nom de produit : une nouvelle soumission d'un produit peut changer à la
  fois de modèle et d'architecture.
- Certaines soumissions sont des systèmes avec plusieurs tentatives ; on prend le résultat publié.
