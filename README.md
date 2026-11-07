# cross-version-transfer

Prototype mathématique du projet Agent Wiki (nom de travail). Question : quand on remplace la version
d'un agent (modèle, prompt, outils), combien de l'information portée par l'historique de l'ancienne
version reste utilisable pour évaluer la nouvelle ?

Idée : l'historique de v n'est pas une preuve pour v'. C'est une **variable auxiliaire** qui partitionne
les tâches (réussies / ratées par v). Quelques probes appariées mesurent v' dans chaque partie
(double échantillonnage, Neyman 1938). Le gain d'efficacité attendu suit la loi finie

    Gain = m_eff / m = 1 / (1 - phi² + phi² · m / n_v)

où phi est la corrélation, tâche par tâche, entre les résultats de v et de v'.

## Contenu

| Chemin | Rôle |
|---|---|
| `src/cvt/estimators.py` | reset, héritage complet, double échantillonnage (bayésien, fréquentiste), **moyenne de modèles (méthode retenue)**, information empruntée |
| `src/cvt/simulate.py` | scénarios synthétiques, tirage des probes, évaluation appariée |
| `tests/test_transfer_estimator.py` | suite synthétique (15 tests, ~5 s) |
| `experiments/` | grilles et validations de la porte G1 |
| `results/` | sorties CSV et journaux |

Aucune dépendance au noyau logiciel (API, MCP, framework d'agents) : `numpy`, `scipy`, `pytest`.

## Lancer

    python -m pytest -q tests
    python experiments/g1_bma_check.py 20
    python experiments/g1_heldout_validation.py 200 400 20 ds_bma proportional

## Méthode retenue (v0.3.2) : `double_sampling_bma`

Deux modèles concourent : M0 « v' ne dépend pas de v » (q1 = q0) et M1 « v' dépend de v »
(double échantillonnage). Ils sont pondérés par leur vraisemblance marginale : l'historique n'est
utilisé que si les probes montrent un lien. Tirage des probes proportionnel. Moyenne lue sur des
tirages Jeffreys, intervalle à 90 % sur des tirages Clopper-Pearson.

## État de la porte G1 (simulation, 7 novembre 2026)

Sur 200 scénarios tirés au hasard, jamais vus pendant la mise au point (20 probes, historique de 500) :

- gain mesuré / gain prédit par la loi finie : médiane 1,00, 90 % des cas entre 0,93 et 1,06 ;
- couverture des intervalles à 90 % : moyenne 91,5 %, minimum 87,7 % (reset : 89,7 %, minimum 81,5 %) ;
- non-dégradation : une seule perte réelle, x0,90, quand v' réussit moins de 10 % des tâches et que les
  versions sont indépendantes (limite connue).

Tout est synthétique : rien n'est encore établi sur des transitions de modèles réelles (porte G2).
