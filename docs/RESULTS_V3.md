# Résultats v3 — confirmation pré-enregistrée sur SWE-bench Verified

Exécutée le 7 novembre 2026 après le gel `prereg-v3` (empreintes vérifiées après exécution :
`docs/PREREG_V3_SHA256.txt`). Script figé : `experiments/v3_swebench_preregistered.py`, R = 400, graine 20261114.
Données : 175 agents publics datés, 500 tâches, 89 transitions de version en ordre temporel réel ; l'écosystème
d'une nouvelle version ne contient que des agents publiés avant elle. Sorties : `results/v3/`.

## Verdict, selon les critères figés (m = 20)

| | Critère | Résultat | Verdict |
|---|---|---|---|
| **H1-v3 — La loi tient** | erreur médiane ≤ 0,20 | **0,036** (corrélation gain observé / prédit 0,989) | soutenue |
| **H5-v3 — Le graphe v3 fait mieux que le prédécesseur seul** | Wilcoxon p < 0,05 ; ≥ 0,95 × reset sur ≥ 90 % ; couverture ≥ 0,88 | **×2,01 contre ×1,53** ; p = 8 × 10⁻¹⁴ ; 97 % ; couverture 0,946 | soutenue |
| **H6-v3 — Pas de perte catastrophique** | pire gain ≥ 0,80 et pire couverture ≥ 0,80 | pire gain ×0,60 ; pire couverture 0,865 | **non soutenue** |
| **H7-v3 — Le prédécesseur ajoute à l'écosystème** | Wilcoxon p < 0,05 | ×2,014 contre ×2,006 ; p = 0,038 | soutenue, effet négligeable |
| **H2-v3 — L'arête de lignée porte une information propre** | Wilcoxon p < 0,05 | φ² médian 0,367 contre 0,316 pour un témoin de même taux ; p = 9 × 10⁻⁷ | soutenue |
| **H4-v3 — Détection de régressions réelles** | alerte ≥ 0,80 ; fausses alertes ≤ 0,10 | 11 régressions réelles : alerte 0,923 avec 30 probes ; fausses alertes 0,008 sur 73 transitions | soutenue |

## Ce que montrent les données

**Le graphe divise le coût d'évaluation par deux.** Avec 20 probes, l'estimateur graphe v3 est aussi précis que
40 probes tirées au hasard (gain médian ×2,01) ; avec 40 probes, autant que 90 (×2,26). Le prédécesseur seul
donne ×1,53. Sur les 85 transitions où la nouvelle version réussit entre 10 et 90 % des tâches, le pire cas est
×1,25 et la couverture des intervalles à 90 % reste entre 0,895 et 0,99.

**La loi d'efficacité est confirmée pour la troisième fois**, sur la source la plus grande et la plus réaliste :
gain observé à 3,6 % près de la loi finie en médiane (90 % des transitions à moins de 11 %).

**Le lien de version est réel, mais le graphe le contient déjà.** Une version partage plus d'information avec son
prédécesseur qu'avec un agent extérieur de même niveau (φ² 0,37 contre 0,32 ; la lignée l'emporte dans 71 % des
cas). Mais une fois la difficulté des tâches vue par tout l'écosystème connue, ajouter le prédécesseur n'apporte
presque rien : différence médiane de gain 0,005, le prédécesseur aide dans 55 % des transitions. H7 passe le seuil
de signification mais son effet est négligeable en pratique ; on ne peut pas en faire un argument.

**La détection de régressions marche sur des régressions réelles.** Les 11 nouvelles versions qui perdent au moins
10 points sont signalées dans 92 % des cas, avec 30 probes en moyenne ; les 73 transitions sans baisse déclenchent
0,8 % de fausses alertes. Les deux régressions les plus faibles (−10 et −10,2 points, autour de 30 % de réussite)
sont détectées 63 à 69 % du temps : c'est la limite de résolution à 80 probes.

**H6 échoue, sur deux cas extrêmes.**
- Une soumission publiée vide (`20260226_mini-v2.0.0_gemini-3-pro-high` : coûts et appels présents, 0 tâche
  résolue sur 500 dans le fichier publié). Toutes les probes valent 0, la garde bascule sur le prédécesseur,
  qui tire l'estimation vers le haut : ×0,60. Le reset, lui, est presque exact.
- RAG + GPT-4 (2,8 % de réussite) : ×0,76, couverture 0,865.
Sans la soumission vide (analyse exploratoire), le pire cas est ×0,76 : H6 échoue quand même. Les versions qui
réussissent moins de 10 % des tâches restent la limite (4 transitions, gain médian ×0,85).

**Dans quel cas le graphe aide moins.** Écosystème de moins de 20 agents (5 transitions, en 2024) : ×1,42 au lieu
de ×2,05 ; le gain croît avec la taille du graphe.

## Comparaison avec les étapes précédentes

| | Transitions | Prédécesseur seul | Graphe | Couverture du graphe |
|---|---|---|---|---|
| v1, modèles (Sheng & Lu), pilote | 30 | ×1,26 | — | — |
| v2, agents AppWorld | 40 | ×1,12 | ×1,40 (v2) | 0,909 |
| v3, agents SWE-bench, ordre temporel réel | 89 | ×1,53 | ×2,01 (v3) | 0,946 |

## Ce que cela autorise à écrire

- Sur 89 versions réelles d'agents de code, évaluées dans l'ordre où elles sont sorties, l'historique du graphe
  d'agents permet d'estimer le taux de réussite d'une nouvelle version avec deux fois moins de tâches, avec des
  intervalles calibrés.
- La loi d'efficacité 1/(1 − φ² + φ²·m/N) prédit le gain à 4 % près, sur trois sources indépendantes.
- La détection séquentielle repère 92 % des régressions réelles de 10 points ou plus, avec 30 tâches.
- Le prédécesseur partage une information propre avec la nouvelle version, mais elle est presque entièrement
  contenue dans le graphe : la valeur vient de l'écosystème, pas de la lignée seule.
- Limites : versions sous 10 % de réussite (et une soumission publiée vide) ; tâches corrélées par dépôt ;
  lignées définies par nom de produit ; travaux proches de type IRT (Agent Psychometrics, Efficient Benchmarking
  in Production, tinyBenchmarks) qui prédisent des résultats sans intervalles calibrés.
