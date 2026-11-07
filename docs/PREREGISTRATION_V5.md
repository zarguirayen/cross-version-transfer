# Pré-enregistrement v5 — généralité sur 9 jeux de tâches jamais analysés

**Statut : figé par le tag git `prereg-v5` avant toute statistique de résultat sur ces données.**
Rédigé le 7 novembre 2026. Analyse figée dans `experiments/v5_general_preregistered.py` et
`experiments/v5_general_data.py` (même commit). Testée à blanc avec `--fake`.

## Pourquoi une v5

Les v3 et v4 portaient sur SWE-bench Python (Verified, Lite). La v5 teste si la méthode vaut ailleurs : sept
langages (Multi-SWE-bench), un découpage de SWE-bench beaucoup plus grand, et un banc multilingue récent. Elle
teste aussi une nouvelle variante, **GREG à ajustement croisé** (`graph_greg_cf`), conçue pour réunir ce qui
manquait en v3 et v4 : ne jamais perdre beaucoup face au reset ET garder des intervalles calibrés.

**GREG croisé** (développé sur données déjà vues) : plan stratifié ; pli A = strates paires, pli B = strates
impaires ; le modèle du graphe ajusté sur un pli prédit les résidus de l'autre (résidus hors échantillon,
cross-PPI de Zrnic & Candès 2024) ; estimation = moyenne des deux modèles + correction pondérée ; variance par
strates appariées ; intervalle de proportion à taille effective corrigée des degrés de liberté (Korn & Graubard
1998, quantiles de Beta de Jeffreys). Motif : en v4, GREG ne perdait jamais mais ses intervalles étaient trop
étroits (couverture minimale 0,625), parce que les résidus étaient mesurés sur les probes qui avaient servi à
ajuster le modèle.

**Développement, sur données déjà vues uniquement** (R = 200, m = 20 ; `results/dev_v5_*`,
`experiments/dev_v5_greg.py`). Gain médian face au reset uniforme (pire cas) ; couverture moyenne (minimum) :

| Données | graphe stratifié `v3_s` | GREG v4 | GREG croisé `greg_cf_s` |
|---|---|---|---|
| SWE-bench Lite, 28 | ×2,22 (×0,88) ; 0,947 (0,845) | ×1,93 (×1,22) ; 0,904 (0,560) | ×1,87 (×1,07) ; 0,940 (0,895) |
| SWE-bench Verified, 89 | ×2,44 (×0,69) ; 0,954 (0,870) | ×2,15 (×1,16) ; 0,913 (0,800) | ×2,11 (×1,08) ; 0,926 (0,000)* |
| AppWorld, 40 | ×1,50 (×0,72) ; 0,928 (0,830) | ×1,41 (×1,01) ; 0,903 (0,690) | ×1,38 (×1,02) ; 0,949 (0,880) |

\* La couverture nulle est la soumission publiée vide (vrai taux exactement 0) : la borne basse de Jeffreys
n'atteint jamais 0. Correction fixée avant le gel : convention de Brown, Cai & DasGupta (2001), borne basse 0 si
aucune probe n'a réussi et borne haute 1 si toutes ont réussi. Revérifié (`dev_v5_lowrate_check.py`) : couverture
1,000 sur ce cas, 0,937 à 0,963 sur les trois autres transitions sous 10 %, gain de ×0,75 à ×1,38 sur ce tirage
(les rapports d'erreur sont très bruités à ces taux).

Compromis connu avant le gel : `v3_s` donne le meilleur gain, mais peut perdre sous 10 % de réussite ;
`greg_cf_s` coûte environ 10 à 15 % de gain mais n'a jamais perdu en développement, avec des intervalles
calibrés, larges sous 10 % de réussite.

## Données (audit de forme seulement)

| Jeu de tâches | Source | Tâches | Soumissions | Transitions |
|---|---|---|---|---|
| C, C++, Go, Java, JavaScript, Rust, TypeScript | `multi-swe-bench/experiments`, `evaluation/<langage>/verified` (commit 6a7d556) | 128 à 428 | 36 à 41 | 3 par langage, 4 en Java (22) |
| SWE-bench test, **hors tâches de Verified et Lite** | `SWE-bench/experiments`, `evaluation/test` | 1 587 | 24 | 9 |
| SWE-bench multilingual | `SWE-bench/experiments`, `evaluation/multilingual` | 301 | 14 | 2 |

Exclus avant toute statistique : Multi-SWE-bench Python (risque de recouvrement avec Verified), SWE-bench
multimodal (totaux seulement), Terminal-Bench (résultats par tâche publiés pour 7 soumissions sur 26).

**Incident d'audit, déclaré :** en vérifiant le format des fichiers Multi-SWE-bench, l'affichage a montré le total
d'une soumission (Go, `20250426_MSWE-agent_Doubao-1.5-thinking` : 18 tâches résolues sur 428). Cette soumission
est la nouvelle version v′ d'une transition. Aucune autre valeur n'a été vue.

**Structure, déclarée :** 21 des 33 transitions sont le même changement de modèle (Doubao-1.5-pro → Doubao-1.5-
thinking) dans 3 échafaudages et 7 langages. Les tests traitent les transitions comme des unités ; une analyse
par jeu de tâches et par famille est rapportée en plus.

## Règles (v5_general_data.py)

Chaque jeu de tâches est un graphe séparé. Lignée : échafaudage fourni par le banc + fournisseur de modèle, ou
même produit. Transition v → v′ : v = soumission la plus récente de la lignée à date strictement antérieure.
Écosystème de v′ : soumissions du même jeu, antérieures à v′, sauf v.

## Protocole

R = 400 rejeux par transition, m ∈ {20 (principal), 40}. Plans uniforme et stratifié tirés indépendamment.
Estimateurs : `reset_u`, `bma_u`, `v3_u`, `reset_s`, `v3_s`, `greg_cf_s`. Graine 20261118.

## Hypothèses confirmatoires (m = 20, 33 transitions)

| | Mesure | Critère |
|---|---|---|
| **H1-v5 — Loi φ²** | \|gain du BMA / loi finie − 1\| | médiane ≤ 0,20 |
| **H8-v5 — Loi du plan stratifié** | \|gain de `reset_s` / loi exacte − 1\| | médiane ≤ 0,20 |
| **H9-v5 — Le graphe qui choisit les tâches fait mieux** | `v3_s` contre `v3_u` | Wilcoxon apparié unilatéral p < 0,05 **et** `v3_s` ≥ 0,95 × reset sur ≥ 90 % **et** couverture moyenne ≥ 0,88 |
| **H10-v5 — Meilleures décisions** | erreur « garder v ou passer à v′ », transitions où \|p′ − p_v\| ≥ 0,05 | Wilcoxon apparié unilatéral, `v3_s` < `reset_u`, p < 0,05 |
| **H12-v5 — Robuste ET calibré** | `greg_cf_s` | pire gain ≥ 0,80 **et** couverture moyenne ≥ 0,88 **et** couverture minimale ≥ 0,80 |

Rapportés sans critère : m = 40 ; résultats par jeu de tâches ; famille Doubao contre autres ; largeurs
d'intervalles ; versions sous 10 %.

## Limites connues avant les résultats

- 33 transitions, dont 21 répliques d'un même changement de modèle.
- Écosystèmes petits pour SWE-bench test (3 à 19 agents).
- Jeux Multi-SWE-bench de 128 à 129 tâches pour C, C++ et Java : strates de 6 à 7 tâches.
