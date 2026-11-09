# Analyse exploratoire : la loi du prédécesseur avec φ² estimé sur les seules tâches testées

**Statut : protocole écrit et commité avant tout calcul de cette analyse (9 novembre 2026). Analyse exploratoire et
rétrospective, pas une confirmation prospective** : les données v3–v6 ont déjà été analysées, mais φ̂² n'y a jamais été
calculé à partir des probes. Une seule exécution ; aucun réglage de l'estimateur après coup.

## Question

En v3–v6, la loi 1/(1 − φ² + φ² m/N) a été comparée au gain observé avec le **vrai** φ², calculé sur toutes les tâches
— une information qu'on n'a pas en production. Avec seulement les 20 tâches testées, la loi annonce-t-elle encore le
gain ? (Le pilote seul l'a testé : 4,4 % d'erreur médiane contre 2,6 % avec le vrai φ².) Seul φ² (transfert depuis le
prédécesseur) est étudié ; η²_G (difficulté du graphe) n'est pas estimé ici, parce qu'avec une probe par strate la
variance intra-strate n'est pas observable.

## Protocole

- Transitions : toutes celles de v3 (89), v4 (28), v5 (33) et v6 (29), m = 20, avec les mêmes chargeurs de données que
  les scripts figés ; chaque transition est appariée à sa ligne de `results/<étude>/replays.csv` (contrôle : mêmes p
  et φ²).
- Pour chaque transition, R = 400 tirages uniformes de 20 tâches sans remise (le plan des estimations à base de
  prédécesseur). Sur ces seules tâches : table des bascules, φ̂ = `estimators.phi_signed`, gain annoncé
  Ĝ = `estimators.gain_law(φ̂², 20, N)`, la fonction utilisée dans les études.
- Cas dégénérés : une marge sans variance sur les 20 tâches (toutes réussies ou toutes échouées, pour l'une des deux
  versions) rend φ̂ indéfini ; `phi_signed` renvoie alors 0 et Ĝ = 1. Leur part est rapportée, et les erreurs sont
  données avec et sans eux.
- Gain observé : `gain_bma` (v3) ou `gain_bma_u` (v4–v6) à m = 20, celui comparé à la loi dans le papier.
- Mesures, rapportées quelle que soit leur valeur :
  1. erreur par transition |médiane_r(Ĝ) / G_obs − 1|, médiane sur les transitions, à côté de l'erreur avec le vrai φ²
     et de la référence « aucun transfert » (Ĝ = 1) ;
  2. erreur d'une seule évaluation : médiane sur toutes les paires (transition, tirage) de |Ĝ / G_obs − 1| ;
  3. biais de φ̂² : médiane sur les transitions de (médiane_r φ̂² − φ²) ;
  4. instabilité : écart interquartile de φ̂² par transition, médiane sur les transitions ;
  5. part des tirages dégénérés.
- Graine 20261109. Script : `experiments/exploratory_phihat.py`. Sorties : `results/exploratory_phihat/`.
