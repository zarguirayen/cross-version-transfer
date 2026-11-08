# Pré-enregistrement v7 — la structure du graphe

**Statut : figé par le tag git `prereg-v7` avant toute statistique de structure sur ces données.**
Rédigé le 8 novembre 2026. Analyse figée dans `experiments/v7_structure_preregistered.py` (même commit).
Testée à blanc avec `--fake` (données sans structure : S1 à S4 non détectées, comme attendu).

## Nature de l'étude : analyse secondaire pré-enregistrée

Aucune source entièrement neuve n'était disponible (τ²-bench : 4 modèles publiés ; SWE-bench multimodal : totaux
seulement ; Terminal-Bench : 7 soumissions). Cette étude réutilise donc les données des confirmations v4 à v6
(SWE-bench Lite, Multi-SWE-bench, SWE-bench test et multilingual, LiveCodeBench, TheAgentCompany), déjà analysées pour
d'autres questions. **Aucune statistique de structure (corrélation de résidus entre agents, voisins appris) n'y a été
calculée avant ce gel.** Le développement s'est fait uniquement sur AppWorld et SWE-bench Verified. C'est une preuve
plus faible qu'une confirmation sur données neuves, et le papier le dira.

## Question

Jusqu'ici, la valeur du graphe pour l'évaluation venait de son échelle de difficulté (la difficulté moyenne vue par
tous les agents) ; la seule arête testée, la lignée, n'ajoutait presque rien (v3, H7). La structure du graphe — qui est
relié à qui — porte-t-elle une information propre ?

## Développement (connu avant le gel, AppWorld et Verified)

- **Structure résiduelle (AppWorld, `dev_v7_residual_structure.py`)** : après un modèle à une dimension (capacité +
  facilité), corrélation moyenne des résidus : même échafaudage +0,04 à +0,06 ; même modèle −0,08 à −0,11 ; sans lien
  −0,09. L'échafaudage, plus que le modèle, façonne l'empreinte d'un agent au-delà de la difficulté.
- **Arêtes apprises (`dev_v7_structure.py`, R = 100)** : GREG croisé avec 10 voisins contre GREG croisé : Verified
  ×2,10 contre ×2,02 (p = 0,0002, rapport médian ≈ 1,02) ; AppWorld ×1,29 contre ×1,26 (p = 0,15). Estimateur fondé
  sur le modèle avec 5 voisins : rapport médian 1,03 (Verified p = 0,008 ; AppWorld p = 0,16). Avec 5 voisins dans
  GREG croisé, rien (5 probes par pli ne suffisent pas à les choisir).

## Hypothèses confirmatoires

| | Mesure | Critère |
|---|---|---|
| **S1 — La structure existe : l'échafaudage** | Multi-SWE-bench, 7 langages, plan croisé initial (3 échafaudages × 9 modèles, 27 agents par langage) : corrélation moyenne des résidus des paires de même échafaudage moins celle des paires sans lien | Wilcoxon unilatéral sur les 7 langages, p < 0,05 |
| **S2 — L'échafaudage compte plus que le modèle** | même mesure : écart « même échafaudage » moins écart « même modèle » | Wilcoxon unilatéral sur les 7 langages, p < 0,05 |
| **S3 — La structure existe : le fournisseur** | LiveCodeBench, un nœud par modèle (variantes d'effort exclues) : paires de même fournisseur contre autres | permutation des étiquettes, p < 0,05 |
| **S4 — La structure aide à évaluer** | 90 transitions v4–v6, plan stratifié, m = 20, R = 200 : gain de GREG croisé avec 10 voisins appris contre GREG croisé | Wilcoxon apparié unilatéral, p < 0,05 |
| **S5 — Mais au second ordre** | rapport médian des gains de S4 | ≤ 1,10 |

Rapportés sans critère : permutations par langage, couvertures, estimateur fondé sur le modèle avec 5 voisins.
Résidus : modèle logit P = capacité + facilité, ajusté sur les agents de l'analyse, pénalité 0,01, 300 itérations.
Graine 20261121.

## Limites connues avant les résultats

- Analyse secondaire (voir plus haut) ; les 21 transitions Doubao de la v5 sont le même changement de modèle.
- Les corrélations de résidus ont un biais négatif commun dû à l'ajustement (visible sur AppWorld) : seules les
  différences entre types de paires sont interprétées.
- Les arêtes typées ne sont testées que là où le plan est croisé (Multi-SWE-bench) ou le fournisseur identifiable
  (LiveCodeBench).
