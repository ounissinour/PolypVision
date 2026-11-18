Mean Dice, all test images

| model                                                      |   Kvasir |   CVC-ClinicDB |   CVC-ColonDB |   ETIS-LaribPolypDB |   CVC-300 |   mean of 5 |
|:-----------------------------------------------------------|---------:|---------------:|--------------:|--------------------:|----------:|------------:|
| PraNet, authors                                            |    0.894 |          0.899 |         0.712 |               0.628 |     0.871 |       0.801 |
| Polyp-PVT, authors (epoch chosen on test)                  |    0.917 |          0.937 |         0.808 |               0.787 |     0.9   |       0.87  |
| Polyp-PVT size_288, ours (re-trained, validation-selected) |    0.899 |          0.889 |         0.782 |               0.706 |     0.871 |       0.829 |

Mean Dice, excluding the 49 flagged near-duplicate candidates

| model                                                      |   Kvasir |   CVC-ClinicDB |   CVC-ColonDB |   ETIS-LaribPolypDB |   CVC-300 |   mean of 5 |
|:-----------------------------------------------------------|---------:|---------------:|--------------:|--------------------:|----------:|------------:|
| PraNet, authors                                            |    0.901 |          0.913 |         0.73  |               0.63  |     0.871 |       0.809 |
| Polyp-PVT, authors (epoch chosen on test)                  |    0.92  |          0.946 |         0.813 |               0.79  |     0.9   |       0.874 |
| Polyp-PVT size_288, ours (re-trained, validation-selected) |    0.907 |          0.886 |         0.784 |               0.708 |     0.871 |       0.831 |

Mean IoU, all test images

| model                                                      |   Kvasir |   CVC-ClinicDB |   CVC-ColonDB |   ETIS-LaribPolypDB |   CVC-300 |   mean of 5 |
|:-----------------------------------------------------------|---------:|---------------:|--------------:|--------------------:|----------:|------------:|
| PraNet, authors                                            |    0.836 |          0.849 |         0.64  |               0.567 |     0.797 |       0.738 |
| Polyp-PVT, authors (epoch chosen on test)                  |    0.864 |          0.889 |         0.727 |               0.706 |     0.833 |       0.804 |
| Polyp-PVT size_288, ours (re-trained, validation-selected) |    0.846 |          0.835 |         0.703 |               0.625 |     0.801 |       0.762 |