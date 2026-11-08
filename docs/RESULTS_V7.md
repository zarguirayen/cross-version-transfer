# Résultats v7 — la structure du graphe (analyse secondaire pré-enregistrée)

Exécutée le 8 novembre 2026 après le gel `prereg-v7` (empreintes vérifiées après exécution). Script figé :
`experiments/v7_structure_preregistered.py`, R = 200, graine 20261121. Sorties : `results/v7/`.
Données des confirmations v4–v6, déjà analysées pour d'autres questions ; aucune statistique de structure calculée
avant le gel ; développement sur AppWorld et SWE-bench Verified seulement.

## Verdict, selon les critères figés

| | Critère | Résultat | Verdict |
|---|---|---|---|
| **S1 — Structure : l'échafaudage** | Wilcoxon sur 7 langages, p < 0,05 | écart médian +0,060 ; positif dans 7/7 ; p = 0,0078 | soutenue |
| **S2 — L'échafaudage compte plus que le modèle** | Wilcoxon sur 7 langages, p < 0,05 | écart médian +0,064 ; 7/7 ; p = 0,0078 | soutenue |
| **S3 — Structure : le fournisseur (LiveCodeBench)** | permutation, p < 0,05 | écart +0,047 ; p = 0,0005 | soutenue |
| **S4 — Les arêtes apprises aident à évaluer** | Wilcoxon apparié, p < 0,05 | ×1,937 contre ×1,955 ; rapport 0,995 ; p = 0,65 | **non soutenue** |
| **S5 — Apport de second ordre** | rapport médian ≤ 1,10 | 0,995 | soutenue (trivialement, vu S4) |

## Ce que montrent les données

- **Le graphe a une structure réelle.** Après avoir retiré la difficulté des tâches et le niveau de chaque agent,
  deux agents qui partagent un échafaudage (MSWE-agent, MagentLess ou MopenHands) échouent encore sur les mêmes
  tâches, dans chacun des 7 langages (permutation significative dans 6 sur 7 ; écart médian +0,060). Partager le
  modèle rapproche aussi, mais moitié moins (écart médian +0,033, positif dans 6 langages sur 7 ; sur AppWorld, aucun
  rapprochement). **C'est l'échafaudage, plus que le modèle, qui façonne l'empreinte d'un agent.** Cela confirme l'atlas de la v2 (changer d'échafaudage efface l'information transférable,
  monter de version de modèle la conserve).
- **Cette structure n'aide pas à estimer une nouvelle version.** Choisir les voisins les plus semblables et s'en
  servir comme prédicteur ne change pas le gain (rapport médian 0,995 ; 1,001 entre 10 et 90 % de réussite). Le petit
  gain vu en développement sur Verified (+2 %) ne se reproduit pas. Pour l'estimation, l'échelle de difficulté du
  graphe contient déjà ce qui compte.
- L'estimateur fondé sur le modèle avec 5 voisins fait plutôt moins bien (×1,793 contre ×1,881), surtout sous 10 % de
  réussite.

## Ce que cela autorise à écrire

Le graphe d'agents n'est pas qu'une échelle de difficulté : ses arêtes typées (même échafaudage, même fournisseur)
portent une information propre, mesurable et reproduite. Mais pour évaluer une nouvelle version, cette structure est
au mieux de second ordre : la valeur d'évaluation du graphe vient de la difficulté des tâches qu'il a apprise.
