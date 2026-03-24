### Benchmark auxiliaire elevation (E102)

Objectif : tester si les 5 RSSI contiennent une signature exploitable de la hauteur du routeur via une tâche auxiliaire supervisée `elevation_class`.

#### Tableau de synthèse

| Métrique | Résultat |
|---|---:|
| Accuracy élévation moyenne | 70.11 % |
| Std accuracy | 0.59 point |
| F1 macro élévation | 0.458 |
| Std F1 macro | 0.019 |
| Accuracy cellule baseline | 16.14 % |
| Accuracy cellule multi-tâche | 16.33 % |
| Gain absolu sur la cellule | +0.19 point |
| F1 macro cellule | 0.146 -> 0.147 |

#### Tableau par campagne

| Campagne | Élévation | Acc. élévation | Acc. cellule baseline | Acc. cellule multi-tâche |
|---|---:|---:|---:|---:|
| exp1_back_right | 0.75 m | 98.6 % | 17.5 % | 18.4 % |
| exp2_front_right | 0.75 m | 87.4 % | 21.1 % | 20.4 % |
| exp3_front_left | 0.75 m | 92.3 % | 15.0 % | 12.9 % |
| exp4_back_left | 0.75 m | 98.2 % | 11.8 % | 15.2 % |
| exp5_ground | 0.00 m | 5.1 % | 17.2 % | 17.1 % |
| exp6_1m50 | 1.50 m | 39.1 % | 14.2 % | 14.2 % |

#### Matrice de confusion agrégée

| Vrai \\ Prédit | 0.00 m | 0.75 m | 1.50 m |
|---|---:|---:|---:|
| 0.00 m | 95 | 1628 | 152 |
| 0.75 m | 59 | 7058 | 383 |
| 1.50 m | 17 | 1124 | 734 |

Lecture : la classe nominale `0.75 m` est bien identifiée, tandis que les hauteurs extrêmes `0.00 m` et `1.50 m` sont souvent confondues avec cette classe intermédiaire. L'information de hauteur existe donc dans les RSSI, mais reste insuffisante pour améliorer nettement la localisation cellule.
