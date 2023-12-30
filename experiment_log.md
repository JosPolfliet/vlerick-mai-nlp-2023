|   precision |   recall |   f1-score |   support | timestamp        | Experiment Name                               |
|------------:|---------:|-----------:|----------:|:-----------------|:----------------------------------------------|
|       0.668 |    0.665 |      0.645 |       547 | 2023-12-30 11:19 | bow                                           |
|       0.67  |    0.682 |      0.657 |       547 | 2023-12-30 11:23 | tf idf                                        |
|       0.646 |    0.671 |      0.654 |       547 | 2023-12-30 11:24 | mean of word embeddings                       |
|       0.563 |    0.569 |      0.544 |       547 | 2023-12-30 11:25 | min and max of word embeddings                |
|       0.553 |    0.555 |      0.532 |       547 | 2023-12-30 11:27 | min, max, lensentence, first word's embedding |
|       0.682 |    0.723 |      0.693 |       547 | 2023-12-30 11:28 | transformer sentence embeddings               |
|       1     |    1     |      1     |       547 | 2023-12-30 11:57 | Transformers dummy                            |