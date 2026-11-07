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
- **Constat** : écart de 0,6 point entre notre taux de régression Mistral MATH (55,8 %) et celui publié par
  Sheng & Lu (55,2 %). Non expliqué ; sans effet sur les conclusions.
