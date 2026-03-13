## Learning curve E102 (train fraction)

- Protocol: stratified random split (test=20%), repeated 10 time(s).
- Fractions explored: 20/40/60/80/100% of the train split.

### KNN

| Train fraction | Acc (mean+-std) | Mean error m (mean+-std) |
|---:|---:|---:|
| 20% | 0.7507 +- 0.0146 | 0.3492 +- 0.0260 |
| 40% | 0.8764 +- 0.0118 | 0.1769 +- 0.0156 |
| 60% | 0.9026 +- 0.0064 | 0.1392 +- 0.0088 |
| 80% | 0.9150 +- 0.0045 | 0.1199 +- 0.0090 |
| 100% | 0.9196 +- 0.0081 | 0.1151 +- 0.0122 |
