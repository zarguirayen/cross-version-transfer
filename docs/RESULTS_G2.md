# Résultats du pilote G2 — analyse pré-enregistrée

Exécutée le 7 novembre 2026 après le gel `prereg-v1` (14 h 44, empreintes dans `docs/PREREG_SHA256.txt`).
Script : `experiments/g2_preregistered.py`, R = 1 000, graine 20261107. Sorties : `results/g2/`.
Données : Sheng & Lu, 30 transitions dans le périmètre (5 paires × 6 benchmarks), 6 hors périmètre.

## Verdict, selon les critères figés

| | Critère pré-enregistré | Résultat | Verdict |
|---|---|---|---|
| **G2** | étendue de φ² ≥ 0,20 et médiane dans [0,05 ; 0,90] | médiane 0,212 ; min 0,049 ; max 0,560 ; étendue 0,511 | **franchie** |
| **H1 — Efficacité** | erreur relative médiane \|observé/prédit − 1\| ≤ 0,20 à m = 20 | **0,026** (m = 10 : 0,042 ; m = 40 : 0,034) ; corrélation observé–prédit 0,97 | **soutenue** |
| H1 secondaire | même mesure avec φ² estimé sur les seules probes | 0,044 à m = 20 | rapporté |
| **H2 — Atlas** | effet paire ou benchmark, Holm α = 0,05 | paire p = 0,012 ; benchmark p = 0,0011 | **soutenue** |
| H2 exploratoire | φ² révision mineure > génération | médianes 0,260 contre 0,202 (sens attendu, sans test) | rapporté |
| **H3 — Décision** | Wilcoxon p < 0,05 **et** regret bma ≤ 1,05 × reset sur ≥ 90 % des transitions | p = 0,0007 (regret total −13 %) mais 25/30 = 83 % | **non soutenue** |
| **H4 — Sécurité** | puissance ≥ 0,80 et fausses alertes ≤ 0,10 à m = 20 ou 40 | m = 40 : puissance 0,671, fausses alertes 0,000 | **non soutenue** |

## Ce que montrent les données

**La loi prédit le gain réel.** Sur les 30 transitions, le gain observé du BMA face au reset vaut 1,23 en
médiane à m = 20 (de 0,98 à 2,25), et le rapport observé/prédit va de 0,91 à 1,17. La loi tient aussi hors
périmètre (Llama-2 → 3 : rapports de 0,93 à 1,03). Avec φ² estimé sur les 20 probes seules, l'erreur médiane
reste de 4,4 % : la loi est utilisable en pratique, sans connaître la vérité.

**Le BMA ne perd presque jamais.** Il bat le reset sur 29 transitions sur 30 ; sa pire perte est de 2 % (Gemma
1.1 → 2 sur GSM8K, où la précision passe de 50 % à 85 % et φ² = 0,08 ; 5 % au contrôle indépendant). Sa
couverture à 90 % va de 88,8 à 95,8 %, contre 88,1 % au minimum pour le reset.

**L'héritage complet est un pari, pas une méthode.** Son erreur médiane est plus faible (x2,5 face au reset
à m = 20), parce que beaucoup de mises à jour changent peu la précision. Mais ses intervalles ne contiennent
la vraie valeur que dans 27 % des cas, et il s'effondre (x0,05) quand la mise à jour change vraiment le modèle.
Le power prior fixe a le même défaut (couverture 37 à 47 %).

**L'atlas.** φ² varie de 0,05 à 0,56 dans le périmètre. Les benchmarks de code et MMLU-Pro conservent le plus
d'information (MBPP : 0,29 à 0,56), GSM8K et GPQA le moins. Gemma 1.1 → 2 (changement de génération et de
taille) a les φ² les plus bas du périmètre (0,05 à 0,29). Hors périmètre, **Llama-2 → Llama-3 efface presque
toute l'information** (φ² ≤ 0,05 sur 5 benchmarks sur 6) et inverse le lien sur GPQA (φ = −0,22) : un saut de
génération remet le compteur à zéro. Un ancien succès sur quatre est perdu sur MMLU-Pro (taux de régression de
24,7 à 29,3 %), même quand la précision globale bouge à peine.

**Contrôles.** Les taux de régression recalculés sur MMLU-Pro retrouvent exactement la fourchette publiée par
Sheng & Lu (24,7–29,3 %) ; écart de 0,6 point sur Mistral MATH (55,8 % contre 55,2 % publié), non expliqué et
sans effet sur nos conclusions. φ² et trois gains ont été recalculés par un autre chemin de code et une autre
graine : φ² identique, gains à 3–5 % près (bruit Monte Carlo).

## Pourquoi H3 et H4 échouent

**H3.** Le BMA réduit le regret total de 13 % (p = 0,0007), mais fait plus de 5 % pire que le reset sur 5
transitions. Ces 5 cas portent sur des regrets de 0,3 à 1 point de précision, où quelques décisions limites
font basculer le ratio. Le critère de robustesse était probablement trop strict pour des regrets aussi petits ;
il n'est pas modifié après coup.

**H4.** Détecter une baisse de 10 points avec 40 probes demande plus que ce qu'apporte l'historique. Le BMA
détecte 67 % des substitutions (le reset 42 %, l'héritage complet 0,1 %), sans aucune fausse alerte.

## Analyse exploratoire (après le gel, hors pré-enregistrement)

Puissance de détection d'une baisse de 10 points selon le nombre de probes (`experiments/g2_exploratory_h4_power.py`,
R = 300) :

| Probes | BMA | Part des transitions ≥ 0,80 | Reset |
|---|---|---|---|
| 20 | 0,24 | 0 % | 0,25 |
| 40 | 0,67 | 3 % | 0,42 |
| 60 | **0,87** | **83 %** | 0,53 |
| 80 | **0,95** | **100 %** | 0,64 |

À confirmer par une nouvelle analyse pré-enregistrée avant d'être revendiqué.

## Ce que cela autorise à écrire

- La loi d'efficacité est validée sur 36 transitions réelles de 6 familles de modèles ouverts, à 3 % près en
  médiane : c'est le résultat central (C2).
- φ² dépend de la paire de versions et du benchmark (C3) ; un saut de génération peut l'annuler.
- Le BMA améliore la décision et la détection sans fausse alerte, mais pas au niveau des critères fixés :
  C4 doit être présenté avec ses chiffres réels, et le budget de probes nécessaire est d'environ 60.
- Limites : modèles et non agents (pattern routage seulement) ; compétence = benchmark ; une seule source de
  données ; dégradation H4 synthétique et uniforme.
