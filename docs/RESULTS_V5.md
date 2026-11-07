# Résultats v5 — généralité sur 9 jeux de tâches jamais analysés

Exécutée le 7 novembre 2026 après le gel `prereg-v5` (empreintes vérifiées après exécution). Script figé :
`experiments/v5_general_preregistered.py`, R = 400, graine 20261118. 33 transitions : Multi-SWE-bench en C, C++,
Go, Java, JavaScript, Rust et TypeScript (22), SWE-bench test hors Verified et Lite (9), SWE-bench multilingual (2).
Sorties : `results/v5/` (dont `descriptive.txt`).

## Verdict, selon les critères figés (m = 20)

| | Critère | Résultat | Verdict |
|---|---|---|---|
| **H1-v5 — Loi φ²** | erreur médiane ≤ 0,20 | 0,058 | soutenue (5e réplication) |
| **H8-v5 — Loi du plan stratifié** | erreur médiane ≤ 0,20 | 0,077 | soutenue (2e réplication) |
| **H9-v5 — Le graphe stratifié fait mieux** | Wilcoxon p < 0,05 ; ≥ 0,95 × reset sur ≥ 90 % ; couverture ≥ 0,88 | ×1,21 contre ×1,12 ; p = 0,0011 ; **76 %** ; couverture 0,920 | **non soutenue** |
| **H10-v5 — Meilleures décisions** | Wilcoxon p < 0,05 | erreur 3,9 % contre 7,6 % (12 transitions) ; p = 0,039 | soutenue |
| **H12-v5 — Robuste et calibré** | pire gain ≥ 0,80 ; couverture moyenne ≥ 0,88 et minimale ≥ 0,80 | **pire gain ×1,08** ; médiane ×1,61 ; couverture 0,950 (min 0,812) | soutenue |

## Ce que montrent les données

**Ce jeu est dominé par des versions très faibles.** 21 des 33 nouvelles versions réussissent moins de 10 % des
tâches (Doubao-1.5-thinking sur Multi-SWE-bench : 1 à 5 %). C'est le régime où les v3 et v4 avaient échoué.

**GREG croisé règle le problème des versions faibles, pour l'estimation.** C'est la seule méthode qui ne perd
jamais face au reset, dans les 33 transitions et les 9 jeux de tâches, avec des intervalles calibrés : gain médian
×1,61 (×1,51 sous 10 % de réussite, ×1,87 au-dessus), couverture 0,950. Le critère « pas de perte catastrophique »,
qui avait échoué en v3, passe désormais avec un estimateur dont les intervalles sont calibrés.

**L'estimateur graphe stratifié reste le meilleur quand les versions ne sont pas extrêmes.** Sur les 12
transitions entre 10 et 90 % : ×1,94 (pire cas ×1,41, couverture 0,95). Sous 10 % : ×1,04, pire cas ×0,58 ; c'est
ce qui fait échouer H9.

**Les décisions de version.** Le graphe stratifié divise par deux l'erreur de décision quand l'écart réel dépasse
5 points (3,9 % contre 7,6 %). GREG croisé, lui, décide moins bien que le reset (9,3 % ; 24 % sur l'ensemble). La
raison est visible dans les données : quand les deux versions réussissent presque rien (0,4 % contre 0,8 %), le
reset, poussé vers le haut par son a priori, déclare toujours la nouvelle version meilleure, et il se trouve que
c'était vrai ; GREG, sans biais, tombe de part et d'autre d'un écart de 0,4 point. Un estimateur non biaisé n'est
pas un bon décideur dans ce régime.

**La loi tient partout.** φ² à 5,8 % près, plan stratifié à 7,7 % près, sur des langages et des bancs nouveaux.

## Règle d'usage qui se dégage (à confirmer avant de la revendiquer)

- Estimer le taux d'une version : GREG croisé (jamais pire que le reset, intervalles calibrés).
- Décider entre deux versions : graphe stratifié.
- Dans les deux cas, choisir les tâches avec le graphe (plan stratifié).

## Limites

- 21 transitions sur 33 sont le même changement de modèle ; par jeu de tâches, GREG croisé dépasse le reset dans
  9 jeux sur 9 et le graphe stratifié dépasse le graphe uniforme dans 7 sur 9.
- Écosystèmes de 3 à 19 agents sur SWE-bench test.
- Incident d'audit déclaré dans le pré-enregistrement (un total vu par accident).
