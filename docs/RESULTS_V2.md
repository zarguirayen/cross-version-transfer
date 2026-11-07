# Résultats v2 — confirmation pré-enregistrée sur des agents (AppWorld)

Exécutée le 7 novembre 2026 après le gel `prereg-v2` (15 h 59, empreintes dans `docs/PREREG_V2_SHA256.txt`).
Script figé : `experiments/v2_appworld_preregistered.py`, R = 400, graine 20261111. Sorties : `results/v2/`.
Données : 14 agents AppWorld (4 architectures × 4 modèles), 168 + 417 tâches, 40 transitions fixées a priori.
Aucune statistique n'avait été calculée sur ces données avant le gel.

## Verdict, selon les critères figés

| | Critère | Résultat | Verdict |
|---|---|---|---|
| **H1-v2 — La loi vaut pour des agents** | erreur médiane ≤ 0,20 | **0,027** | soutenue |
| **H5 — Le graphe fait mieux que le prédécesseur seul** | Wilcoxon p < 0,05 ; ≥ 0,95 × reset sur ≥ 90 % ; couverture moyenne ≥ 0,88 | gain médian ×1,40 contre ×1,12 ; p < 0,0001 ; **90 % tout juste** (36/40) ; couverture 0,909 | soutenue, à la limite |
| **H2-v2 — φ² dépend du type de changement** | Kruskal-Wallis p < 0,05 | p = 0,0026 | soutenue |
| **H4-v2 — Détection séquentielle** | puissance ≥ 0,80, ≤ 60 probes, fausses alertes ≤ 0,10 | 0,859 avec 52 probes ; fausses alertes 0,001 | soutenue (voir réserve) |

## Ce que montrent les données

**La loi tient pour des agents.** Comme sur les modèles, le gain observé du BMA suit la loi finie à 2,7 % près en
médiane, sur 40 transitions d'agents qui combinent changements de modèle, d'architecture et de famille.

**L'historique d'une version d'agent survit à une montée de version du modèle, beaucoup moins à un changement
d'architecture.** φ² médian : 0,30 pour GPT-4 Turbo → GPT-4o à architecture fixe ; 0,08 pour un changement
d'architecture à modèle fixe ; 0,045 pour un changement de famille de modèle. Réécrire l'orchestration d'un
agent efface presque autant d'information prédictive que changer de fournisseur de modèle.

**Le graphe récupère ce que la lignée perd.**

| Type de transition | φ² médian | Gain médian, prédécesseur seul | Gain médian, graphe | Graphe meilleur |
|---|---|---|---|---|
| A — montée de version du modèle | 0,30 | ×1,47 | ×1,51 | 88 % |
| B — changement d'architecture | 0,08 | ×1,10 | ×1,20 | 60 % |
| C — changement de famille de modèle | 0,045 | ×1,06 | **×1,49** | **100 %** |

Quand le prédécesseur n'apprend plus rien (type C), l'historique des autres agents du graphe donne encore un
gain de ×1,2 à ×1,8. C'est le même phénomène que Llama-2 → Llama-3 en développement.

**Limite nette : les versions qui réussissent moins de 10 % des tâches.**

| Taux de réussite de la nouvelle version | Transitions | Gain médian du graphe | Pire cas | Couverture moyenne (min) |
|---|---|---|---|---|
| ≥ 10 % | 34 | ×1,43 | ×1,05 | 0,923 (0,860) |
| < 10 % | 6 | ×0,69 | ×0,38 | 0,828 (0,677) |

Les 4 échecs de H5 sont tous dans la seconde ligne (PlanExec avec Llama-3 ou DeepSeekCoder, 0,7 à 8,9 % de
réussite) : avec presque aucun succès parmi 20 probes, la régression logistique extrapole mal. Le BMA y résiste
mieux (pire cas ×0,75). Règle d'usage à confirmer dans une v3 : basculer sur le BMA quand les probes contiennent
moins de 2 succès.

**Réserve sur H4.** Les 11 agents testés ont des taux entre 0,21 et 0,49 ; un bruit symétrique fait alors monter
leur taux, ce qui rend le contrôle des fausses alertes pré-enregistré trop facile. Contrôle exploratoire avec un
bruit qui ne change pas le taux moyen (`experiments/v2_exploratory_h4_null.py`) : fausses alertes 0,3 % quand
5 % des tâches changent de résultat, 0,7 % quand 10 % changent. La conclusion tient. Agents avec un taux > 0,5
non testés (aucun dans AppWorld).

**Analyses secondaires.** À m = 40 : graphe ×1,37, BMA ×1,08. Prior P(lien) = 0,9 : ×1,13 (contre ×1,12) —
amélioration négligeable sur les agents. Écosystème complet (agents du même modèle inclus) : ×1,36, pas mieux que
l'écosystème prudent.

## Ce que cela autorise à écrire

- La loi d'efficacité est confirmée sur deux sources indépendantes, des modèles (36 transitions) et des agents
  (40 transitions), avec une erreur médiane de 3 %.
- Utiliser tout le graphe d'agents, et pas seulement le prédécesseur, augmente le gain médian de ×1,12 à ×1,40 sur
  des données jamais vues, à condition que la nouvelle version réussisse au moins 10 % des tâches.
- La part d'information qui survit dépend du type de changement : forte pour une montée de version du modèle,
  faible pour un changement d'architecture ou de famille.
- La détection séquentielle repère une dégradation de 10 points dans 86 % des cas avec 52 probes en moyenne,
  sans fausse alerte notable.
- Limites : taux < 10 % ; une seule campagne AppWorld (pas d'ordre temporel entre agents) ; tâches regroupées par
  scénario ; dégradations et bruit synthétiques pour H4.
