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

## Résultats (exécution unique, 9 novembre 2026)

Une première exécution s'est arrêtée sur le contrôle d'appariement avant tout calcul de résultat : `replays.csv` de v4
contient aussi les lignes du sous-ensemble secondaire « hors Verified ». Le script a été restreint à l'ensemble
principal (`tasks == "toutes"`), puis exécuté une fois. Sorties : `results/exploratory_phihat/summary.txt`.

| | n | erreur par transition, φ̂² | vrai φ² | « aucun transfert » | erreur d'une évaluation | biais de φ̂² | IQR de φ̂² | tirages dégénérés |
|---|---|---|---|---|---|---|---|---|
| tout | 179 | 5,3 % | 4,4 % | 31 % | 20 % | +0,006 | 0,27 | 14 % |
| v3 | 89 | 4,7 % | 3,8 % | 35 % | 22 % | +0,011 | 0,28 | 5 % |
| v4 | 28 | 4,4 % | 3,8 % | 32 % | 19 % | +0,010 | 0,27 | 7 % |
| v5 | 33 | 10,0 % | 5,5 % | 18 % | 16 % | −0,021 | 0,003 | 55 % |
| v6 | 29 | 6,1 % | 6,1 % | 36 % | 25 % | +0,007 | 0,29 | 4 % |

Lecture : en moyenne sur les tirages, la loi avec φ̂² retrouve le gain presque aussi bien qu'avec le vrai φ² ; une seule
évaluation de 20 tâches donne un φ̂² trop instable pour prévoir le budget à ±20 % près.
