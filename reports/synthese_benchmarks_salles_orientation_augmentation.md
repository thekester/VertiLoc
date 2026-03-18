# Synthese Benchmarks — Salles, Combinaison, Features, Algorithmes, Augmentation

## 1. Perimetre

Cette synthese regroupe les principaux resultats disponibles dans le depot sur :

- la localisation par cellule selon les salles (`D005`, `E101`, `E102`) ;
- la combinaison multi-salles ;
- la prediction d'orientation sur `E101` et `E102` ;
- les features utilises pour l'orientation ;
- la comparaison par algorithmes ;
- les essais de data augmentation generative.

Note de lecture :

- `F1 macro` = moyenne simple du score `F1` calcule separement pour chaque classe ;
- contrairement a l'accuracy globale, il donne le meme poids a chaque classe et penalise donc les modeles qui reussissent bien certaines classes mais ratent fortement les autres.
- `room-aware` = le modele connait explicitement la salle au moment de l'inference ;
- `deg` = degres angulaires ; `m` = metres.

## 2. Vue d'ensemble des salles

- `D005` : salle plus simple, utilisee surtout pour la localisation classique a 2 m / 4 m.
- `E101` : salle intermediaire, avec campagnes de distance et campagnes circulaires `back/front/left/right`.
- `E102` : salle la plus riche experimentalement, avec 4 modes d'orientation (`back_right`, `front_right`, `front_left`, `back_left`) et 2 campagnes d'elevation.

En pratique :

- la localisation **room-aware** multi-salles est tres forte ;
- la generalisation **hors salle** ou **hors campagne** reste tres faible ;
- l'orientation est un probleme nettement plus difficile que la cellule.

## 3. Combinaison des salles — localisation cellule

Le meilleur regime global observe est le protocole **room-aware** sur la combinaison des salles, c'est-a-dire quand l'apprentissage sait dans quelle salle on se trouve.

Top algorithmes sur la combinaison multi-salles (`benchmark_summary_stats.csv`) :

| Rang | Algorithme | Accuracy cellule moyenne | Erreur moyenne |
|---|---|---:|---:|
| 1 | `LDA+KNN (room-aware, cell)` | `0.9806` | `0.0138 m` |
| 2 | `KNN (room-aware, cell)` | `0.9800` | `0.0169 m` |
| 3 | `KNN distance+scaled (room-aware, cell)` | `0.9785` | `0.0178 m` |
| 4 | `GradientBoosting (room-aware, cell)` | `0.9782` | `0.0185 m` |
| 5 | `HistGradientBoosting (room-aware, cell)` | `0.9779` | `0.0181 m` |

Lecture :

- les methodes type `KNN` restent tres competitives ;
- `LDA+KNN` est le meilleur score global observe sur la combinaison des salles ;
- les ensembles d'arbres (`RandomForest`, `ExtraTrees`, `GradientBoosting`) restent tres solides mais ne depassent pas le meilleur `KNN/LDA+KNN`.

## 4. Robustesse par salle / generalisation

### 4.1 E102 — localisation cellule hors campagne

Le protocole `LOCO E102` (`loco_e102_summary.json`) montre une forte chute de performance quand on tient une campagne complete hors apprentissage :

- `RandomForest` : accuracy moyenne d'environ `0.0199`
- `KNN` : accuracy moyenne d'environ `0.0162`
- `ExtraTrees` : accuracy moyenne d'environ `0.0167`

Conclusion :

- la cellule se localise tres bien **dans le meme contexte de salle/campagne** ;
- la generalisation a une campagne non vue reste aujourd'hui tres limitee.

### 4.2 E101 circulaire — orientation tenue hors mode

Le benchmark `loco_e101_circulaire.csv` est egalement tres dur si on retire une orientation complete a l'entrainement :

- meilleur algo moyen : `HistGB`, accuracy moyenne d'environ `0.0181`

Conclusion :

- predire une orientation jamais vue en apprentissage ne fonctionne pas de maniere exploitable ;
- il faut des exemples de chaque mode cible dans les donnees d'entrainement.

## 5. Features utilises pour l'orientation

### 5.1 Features brutes

Le modele d'orientation part des 5 mesures RSSI :

- `Signal`
- `Noise`
- `signal_A1`
- `signal_A2`
- `signal_A3`

### 5.2 Features derivees explorees

Plusieurs familles ont ete testees sur `E102` :

- `raw`
- `signal_vs_ant`
- `raw_plus_deltas`
- `engineered`

La famille la plus utile est :

- `signal_vs_ant`

avec 4 features seulement :

- `signal_minus_a1`
- `signal_minus_a2`
- `signal_minus_a3`
- `noise_minus_signal`

Interpretation :

- les differences entre le signal global et chaque antenne portent plus d'information d'orientation que les deltas inter-antennes seuls ;
- les features trop riches ou trop nombreuses degradent souvent la stabilite.

## 6. Orientation par salle

### 6.1 E101

Le modele circulaire sur `E101` obtient (`orientation_circular_metrics_e101.json`) :

- accuracy orientation : `0.5227`
- F1 macro : `0.5259`
- erreur angulaire moyenne : `56.4 deg`

### 6.2 E102

Baseline `raw + RF` (`orientation_circular_metrics_e102_raw.json`) :

- accuracy orientation : `0.3907`
- F1 macro : `0.3924`
- erreur angulaire moyenne : `66.3 deg`

Meilleur modele ponctuel `signal_vs_ant + RF` (`orientation_circular_metrics_e102_signal_vs_ant_rf.json`) :

- accuracy orientation : `0.4487`
- F1 macro : `0.4492`
- erreur angulaire moyenne : `66.2 deg`

### 6.3 E102 — evaluation plus robuste

Le benchmark multi-seeds (`orientation_feature_benchmark_e102_summary.csv`) confirme que `signal_vs_ant + RF` reste le meilleur compromis, mais avec un gain modeste :

- `signal_vs_ant_rf` : accuracy moyenne `0.358`
- `raw_rf` : accuracy moyenne `0.343`

Le protocole plus strict par lignes tenues hors entrainement (`orientation_line_generalization_e102_ranking.csv`) preserve aussi cet avantage :

- `signal_vs_ant_rf` : `0.3659`
- `raw_rf` : `0.3181`

### 6.4 Combinaison E101 + E102

Le modele combine `E101+E102` a aussi ete teste pour l'orientation circulaire.

Avec le modele combine de reference (`orientation_circular_model.joblib`), on observe :

- accuracy orientation : `0.287`
- F1 macro : `0.282`
- erreur angulaire moyenne : `70.5 deg`

Interpretation :

- melanger `E101` et `E102` ne produit pas un meilleur modele d'orientation global ;
- au contraire, la combinaison des deux salles degrade fortement par rapport a `E101` seul et reste inferieure a `E102` seul dans les splits simples ;
- cela suggere un **shift inter-salles important** pour l'orientation, plus fort encore que pour la cellule.

Conclusion orientation :

- `E101` est plus facile que `E102` ;
- `E101+E102` degrade encore davantage la prediction d'orientation ;
- sur `E102`, le meilleur candidat reste `signal_vs_ant + RandomForest` ;
- l'orientation reste toutefois bien plus difficile que la cellule.

## 7. Performance par algorithmes — orientation

Sur `E102`, les meilleurs couples algo + features observes sont :

| Rang | Configuration | Accuracy moyenne | Erreur angulaire moyenne |
|---|---|---:|---:|
| 1 | `signal_vs_ant + RF` | `0.358` | `76.8 deg` |
| 2 | `signal_vs_ant + ExtraTrees` | `0.351` | `77.3 deg` |
| 3 | `raw + RF` | `0.343` | `76.1 deg` |
| 4 | `raw_plus_deltas + RF` | `0.332` | `77.8 deg` |

Lecture :

- `RF` domine encore sur l'orientation ;
- `ExtraTrees` est proche mais legerement en dessous ;
- `HistGradientBoosting` n'a pas depasse `RF` dans les essais actuels.

## 8. Multitache cellule + orientation

Le comportement du multitache depend clairement de la salle consideree.

### 8.1 E101 seul

Sur `E101` (`e101_orientation_multitask_w005_summary.json`), le multitache aide legerement l'orientation :

- orientation seule : accuracy moyenne `0.6610`
- multitache : accuracy moyenne `0.6660`
- delta accuracy : `+0.0050`
- delta erreur angulaire : `-0.02 deg`

La tete cellule multitache reste faible :

- accuracy cellule multitache : environ `0.131`

Interpretation :

- sur `E101`, la tache cellule apporte un petit biais structurel utile a l'orientation ;
- le gain reste faible mais il est coherent et positif.

### 8.2 E102 seul

Sur `E102` (`e102_orientation_multitask_w005_summary.json`), le multitache n'apporte pas de gain net :

- orientation seule : accuracy moyenne `0.5356`
- multitache : accuracy moyenne `0.5352`
- delta accuracy : `-0.0004`
- delta erreur angulaire : `-0.19 deg` (negligeable)

La tete cellule multitache reste faible :

- accuracy cellule multitache : environ `0.161`

Interpretation :

- sur `E102`, la supervision cellule est globalement neutre pour l'orientation ;
- elle ne deteriore presque plus si le poids cellule reste tres faible (`0.05`), mais n'apporte pas de benefice clair.

### 8.3 Combinaison E101 + E102

Sur la combinaison `E101+E102` (`e101_e102_orientation_multitask_w005_summary.json`), le multitache redevient legerement negatif :

- orientation seule : accuracy moyenne `0.3747`
- multitache : accuracy moyenne `0.3700`
- delta accuracy : `-0.0047`
- delta erreur angulaire : `+0.39 deg`

La tete cellule multitache est encore plus faible :

- accuracy cellule multitache : environ `0.089`

Interpretation :

- la combinaison des deux salles introduit un shift inter-salles fort ;
- dans ce cas, la tete cellule n'aide plus la tete orientation et semble ajouter de la variance.

### 8.4 Conclusion multitache

- `E101` : leger gain du multitache sur l'orientation
- `E102` : effet quasi neutre
- `E101+E102` : leger effet negatif

Conclusion generale :

- la supervision cellule peut aider quand la structure radio est relativement homogene (`E101`) ;
- elle ne suffit pas a compenser l'heterogeneite inter-salles ;
- le vrai levier principal reste la qualite des features et la stabilite du protocole de collecte.

### 8.5 Tableau comparatif compact

| Configuration | Accuracy orientation seule | Accuracy multitache | Delta accuracy | Erreur angulaire seule | Erreur angulaire multitache | Delta erreur |
|---|---:|---:|---:|---:|---:|---:|
| `E101` | `0.6610` | `0.6660` | `+0.0050` | `42.57 deg` | `42.55 deg` | `-0.02 deg` |
| `E102` | `0.5356` | `0.5352` | `-0.0004` | `56.50 deg` | `56.31 deg` | `-0.19 deg` |
| `E101+E102` | `0.3747` | `0.3700` | `-0.0047` | `56.17 deg` | `56.56 deg` | `+0.39 deg` |

## 9. Data augmentation generative

Les essais de data augmentation sont resumes dans `generative_trials_summary.csv` et `generative_trials_stats.csv`.

Constat principal :

- **aucune configuration ne montre de gain net en accuracy** sur les essais disponibles ;
- la meilleure tendance pratique est obtenue avec `CGAN`, jamais avec `CycleGAN`.

Rappel conceptuel :

- `CGAN` genere directement des echantillons conditionnes par une classe ou une campagne cible ; il sert donc surtout a **ajouter des exemples synthetiques** dans le meme domaine.
- `CycleGAN` apprend plutot une **traduction entre deux domaines** (par exemple une campagne source vers une campagne cible), ce qui est plus ambitieux mais aussi plus instable quand la structure radio fine doit etre preservee.

Cas les plus favorables observes :

- `cgan / exp5 / ratio 0.1 / 40 epochs / 20 synth.` :
  - `delta_acc = -0.0053`
  - `delta_err = -0.0136 m`
- `cgan / exp1 / ratio 0.1 / 40 epochs / 20 synth.` :
  - `delta_acc = 0.0000`
  - `delta_err = -0.0052 m`

Les essais `CycleGAN` degradent clairement l'accuracy :

- pertes de `-0.0347` a `-0.0640` selon les cibles testees.

Conclusion augmentation :

- **meilleure augmentation actuelle = CGAN**, mais les gains restent faibles ou nuls ;
- `CycleGAN` n'est pas recommande dans l'etat actuel, car la traduction de domaine semble trop deformer la structure RSSI utile a la localisation.

## 10. Recommandations pratiques

### Pour la localisation cellule

- meilleur choix global multi-salles : `LDA+KNN (room-aware, cell)`
- alternatives robustes : `KNN`, `KNN distance+scaled`, `GradientBoosting`, `HistGradientBoosting`

### Pour l'orientation

- meilleur choix actuel : `signal_vs_ant + RandomForest`
- a conserver comme baseline orientation de reference

### Pour l'augmentation

- conserver `CGAN` comme seule piste encore defendable ;
- abandonner `CycleGAN` pour le moment ;
- ne pas compter sur l'augmentation comme levier principal tant que les features restent pauvres.

## 11. Message de synthese

Le depot montre un contraste fort :

- **la localisation cellule dans une salle connue est excellente** ;
- **la generalisation hors contexte et la prediction d'orientation restent difficiles**.
- **la generalisation circulaire pour l'orientation existe mais reste partielle** : le modele capture surtout des voisinages angulaires coherents, sans fournir encore une orientation fine robuste dans tous les contextes ;
- **la generalisation spatiale au niveau des zones du tableau est en revanche robuste** : un decoupage macro du tableau fournit une sortie plus stable et plus exploitable que la cellule H3 exacte lorsque les erreurs restent locales.

La meilleure strategie actuelle est donc :

1. utiliser les approches `room-aware` pour la cellule ;
2. utiliser `signal_vs_ant + RF` pour l'orientation ;
3. considerer la data augmentation comme secondaire ;
4. investir prioritairement dans de nouvelles variables explicatives (information temporelle, mesures plus riches, ou capteurs additionnels) si l'objectif devient une orientation robuste en conditions reelles.
