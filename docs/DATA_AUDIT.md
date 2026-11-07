# Audit des données — semaine 1 (7 novembre 2026)

Audit de **forme** uniquement : volumes, colonnes, accès, droits. Aucune statistique de résultat n'a été
calculée sur les données réelles avant le gel du pré-enregistrement (`docs/PREREGISTRATION.md`).

## Source principale — Sheng & Lu (arXiv 2608.13607)

- Dépôt : https://github.com/jiashengsally/llm-regression-signals, commit `ce974d7bedebe3b12af6ba354b0eebabc9d3dd28` (11 août 2026).
- Contenu utile : `results/results_{paire}_{benchmark}.csv`, une ligne par question, colonnes `old_correct` et
  `new_correct` (binaires, vérifié sur toutes les lignes), plus des signaux de divergence non utilisés ici.
- Décodage glouton (température 0) selon l'article ; notation : correspondance d'option (QCM), réponse
  normalisée (GSM8K, MATH), tests unitaires (code).

| Paire (ancien → nouveau) | Type de changement | MMLU-Pro | GPQA | GSM8K | MATH | HumanEval | MBPP |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct → Qwen3-8B | génération | 3 000 | 448 | 1 319 | 5 000 | 164 | 500 |
| Meta-Llama-3-8B-Instruct → Llama-3.1-8B-Instruct | révision mineure | 3 000 | 448 | 1 319 | 5 000 | 164 | 500 |
| Mistral-7B-Instruct v0.2 → v0.3 | révision mineure | 3 000 | 448 | 1 319 | 5 000 | 164 | 500 |
| Qwen2-7B-Instruct → Qwen2.5-7B-Instruct | génération | 3 000 | 448 | 1 319 | 5 000 | 164 | 500 |
| gemma-1.1-7b-it → gemma-2-9b-it | génération (et taille) | 3 000 | 448 | 1 319 | 5 000 | 164 | 500 |
| Llama-2-7b-chat → Meta-Llama-3-8B-Instruct | génération, hors périmètre de l'article | 3 000 | 448 | 1 319 | 5 000 | 164 | 500 |

Total : 36 transitions, 62 586 paires (question, transition). Périmètre principal : les 30 transitions des
5 premières paires ; Llama-2 → 3 sert d'analyse secondaire.

**Limites constatées**
- Pas de fichier de licence : usage de recherche avec citation ; **pas de redistribution** des données sans
  l'accord des auteurs (à demander).
- Pas d'identifiant de question ni de catégorie MMLU-Pro : la « compétence » est le benchmark (6 niveaux).
- Le classement « révision mineure / génération » est le nôtre, fixé dans le pré-enregistrement.
- Ce sont des modèles, pas des agents : ces données couvrent le pattern **routage**.

## Pattern agent avec outils — AppWorld

- Les sorties d'expériences des agents de l'article AppWorld (ReAct, PlanExec, FullCodeRefl, IPFunCall ×
  plusieurs modèles, dont GPT-4 Turbo et GPT-4o) sont téléchargeables : `appworld download experiment-outputs`
  ; évaluation par tâche dans `experiments/outputs/{experiment}/evaluations/`. Autres participants : dépôt
  `stonybrooknlp/appworld-leaderboard`.
- Paires de versions utilisables : GPT-4 Turbo → GPT-4o à scaffold fixé (changement de modèle) ; changement de
  scaffold à modèle fixé (changement de prompt / orchestration).
- **À faire** : tester le téléchargement (le serveur de données n'a pas encore été essayé depuis l'espace de
  travail) et vérifier que les mêmes tâches sont évaluées par chaque agent.

## Pattern évaluateur–optimiseur

- Besoin : verdicts par item de plusieurs versions d'un même juge, sur des réponses dont la qualité est établie
  indépendamment (type RewardBench, JudgeBench).
- **À faire** : Hugging Face n'est pas joignable depuis l'espace de travail ; ces données peuvent être
  récupérées depuis l'ordinateur de Rayen, ou en générant les verdicts avec des juges ouverts.

## Open LLM Leaderboard (détails par question)

- Non joignable depuis l'espace de travail (Hugging Face bloqué) ; accès partiellement restreint côté Hugging
  Face. Source secondaire, non nécessaire pour le pilote G2 grâce aux données de Sheng & Lu.

## Actions

1. Écrire à Sheng & Lu : citation, droit de redistribuer les résultats dérivés, éventuels identifiants de
   question et catégories MMLU-Pro.
2. Tester `appworld download experiment-outputs`.
3. Choisir la source du pattern évaluateur.
