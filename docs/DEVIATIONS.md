# Écarts au pré-enregistrement

Chaque entrée : date, ce qui change, pourquoi, effet attendu sur les conclusions.

## Écarts entre la spécification v0.3.1 et le pré-enregistrement (avant gel)

- **Méthode** : la spécification v0.3.1 prévoyait un double échantillonnage avec garde-fou. Les tests de la
  semaine 1 ont conduit à la moyenne de modèles (`double_sampling_bma`), validée hors échantillon sur 200
  scénarios synthétiques. Choix fait avant toute donnée réelle.
- **Critère de couverture de G1** : fixé à ≥ 88 % partout. Sur les 200 scénarios hors échantillon, la méthode
  descend à 87,7 % (1 % des scénarios sous 88 %, compatible avec le bruit de mesure), contre 81,5 % pour le
  reset. Considéré comme atteint ; noté ici par transparence.
- **Limite connue** : perte réelle de 10 % face au reset quand v′ réussit moins de 10 % des tâches et que les
  versions sont indépendantes (1 scénario sur 200).
- **H3** simplifiée en décision binaire (garder v ou passer à v′) ; **H4** fixée à une dégradation de 10 points.
- **Baselines** PPI++, commensurate prior, power prior empirique : reportées en analyses secondaires.

## Après le gel

- **7 novembre 2026, analyse exploratoire ajoutée** : courbe de puissance H4 pour m = 60 et 80
  (`experiments/g2_exploratory_h4_power.py`, graine 20261108). Motif : H4 a échoué à m ≤ 40 ; on cherche le
  budget nécessaire. Statut : exploratoire, à re-tester dans un nouveau pré-enregistrement avant toute
  revendication. Les critères et l'analyse principale ne sont pas modifiés.
- **7 novembre 2026, après le gel v2, analyse exploratoire** : fausses alertes de H4-v2 sous un bruit qui ne
  change pas le taux moyen (`experiments/v2_exploratory_h4_null.py`). Motif : le bruit symétrique pré-enregistré
  fait monter le taux des agents testés (tous sous 0,5), ce qui rendait le contrôle trop facile. Résultat :
  0,3 % et 0,7 % de fausses alertes. Le verdict pré-enregistré n'est pas modifié.
- **Constat v2** : les 4 échecs de H5 concernent des versions à moins de 10 % de réussite. La règle de repli
  (BMA si moins de 2 succès parmi les probes) est une proposition, à confirmer dans une v3 sur données neuves.
- **7 novembre 2026, développement v3 avant le gel `prereg-v3`** (données déjà vues uniquement) : scripts
  `dev_v3_estimator.py`, `dev_v3_rules.py`, `dev_v3b_estimator.py` (prior empirique, arrêté à 23/40 transitions
  faute de gain net, abandonné), `dev_h7_appworld.py`. Toutes les variantes essayées sont listées dans
  PREREGISTRATION_V3.md.
- **7 novembre 2026, après le gel v3, constat sur les données** : la soumission
  `20260226_mini-v2.0.0_gemini-3-pro-high` est publiée avec resolved = false pour les 500 tâches (coûts et appels
  présents). Elle reste dans l'analyse principale, conformément à la règle figée. Analyse de sensibilité
  exploratoire sans elle (2 transitions en moins) : graphe ×2,03, pire cas ×0,76, H7 p = 0,022, H2 inchangée.
  Aucun verdict pré-enregistré n'est modifié ; H6 échoue dans les deux cas.
- **Analyses descriptives v3** : `experiments/v3_descriptive.py` (prévues comme « rapportées sans critère »).
- **7 novembre 2026, développement v4 avant le gel `prereg-v4`** (données déjà vues) : `dev_v4_design.py`
  (plan stratifié ; deux passes, la seconde avec GREG), `dev_v4_law.py`. La variance de GREG a été corrigée une
  fois en développement (tirage uniforme : variance du résidu moyen ; plan stratifié : strates appariées) ;
  plancher de variance fixé à un quart de la variance binomiale de Jeffreys. Toutes les variantes sont listées
  dans PREREGISTRATION_V4.md.
- **Après le gel v4** : aucun écart ; tests ajoutés (`tests/test_v4_design.py`), sans effet sur l'analyse.
- **7 novembre 2026, développement v5 avant le gel `prereg-v5`** : `dev_v5_greg.py`, `dev_v5_lowrate_check.py`.
  Une correction fixée avant le gel (convention de Jeffreys pour 0 ou m succès), décrite dans
  PREREGISTRATION_V5.md. Terminal-Bench et SWE-bench multimodal écartés avant toute statistique (données
  incomplètes).
- **Après le gel v5, analyse exploratoire** : erreurs de décision de GREG croisé selon l'écart réel (inline,
  reprise dans RESULTS_V5.md). Aucun verdict modifié.
- **7 novembre 2026, v6** : avant le gel, normalisation des noms de tâches de TheAgentCompany (suffixe « -image »,
  deux conventions) et exclusion de Terminal-Bench, Multi-SWE-bench Python, SWE-bench multimodal. Après le gel :
  aucun écart ; un test ajouté (`test_greg_cf_covers_zero_rate_and_is_calibrated`), sans effet sur l'analyse.
- **8 novembre 2026, v7** : avant le gel, exclusion des variantes d'effort de LiveCodeBench (un nœud par modèle),
  décidée à l'audit des noms. Après le gel : aucun écart.
- **9 novembre 2026, v8** : avant le gel, déchiffrement des 13 nouveaux classements AppWorld avec le paquet officiel,
  audit de forme seulement ; IBM-HCL-GP n'a pas de fichier d'évaluation, succès lu dans les rapports par tâche (règle
  vérifiée identique sur 4 agents) ; un répertoire « _OLD_54.8_hardcoded » contenu dans un paquet LARA est ignoré.
  Développement de l'estimateur d'échéance sur les 14 agents de l'article : une première version, sans incertitude sur
  la capacité, couvrait 0,84 ; corrigée avant le gel. Après le gel : analyse exploratoire
  `experiments/v8_exploratory_async.py` (diagnostic de l'échec d'A2–A3 et variante « difficulté seule ») ; aucun
  verdict modifié.
- **9 novembre 2026, après v8, développement pour une v9 prospective** (`experiments/dev_v9_deadline.py`, données v8
  déjà vues, non rapporté comme preuve) : à l'échéance, imputer par la difficulté seule (γ = 0) fait mieux que la
  correction v8 (rapport RMSE 1,30 contre 0,88 ; couverture 0,856 contre 0,767, min 0,26) ; un γ appris sur les 5
  agents les plus récents fait moins bien (1,04 ; 0,798). À confirmer seulement sur des agents publiés après le
  2026-11-04.
- **Constat** : écart de 0,6 point entre notre taux de régression Mistral MATH (55,8 %) et celui publié par
  Sheng & Lu (55,2 %). Non expliqué ; sans effet sur les conclusions.
