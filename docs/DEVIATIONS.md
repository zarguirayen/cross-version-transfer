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

(aucun)
