# Feature Selection Summary

## 1. Feature Importance Metrics

|                               |   Correlation (Abs) |   Mutual Information |   Tree Importance |
|:------------------------------|--------------------:|---------------------:|------------------:|
| risk_flags_count              |          0.240634   |          0.0187653   |       0.154932    |
| transaction_hour              |          0.137103   |          0.0139849   |       0.108759    |
| device_trust_score            |          0.132951   |          0.0146433   |       0.107558    |
| late_night_x_low_trust        |          0.298894   |          0.01899     |       0.101884    |
| velocity_last_24h             |          0.109981   |          0.00745603  |       0.0747283   |
| device_trust_bin_very_low     |          0.184306   |          0.0112858   |       0.0600661   |
| foreign_transaction           |          0.174984   |          0.0081273   |       0.0580089   |
| hour_cos                      |          0.124536   |          0.00893059  |       0.050606    |
| location_mismatch             |          0.163459   |          0.00606242  |       0.0496086   |
| late_night_flag               |          0.147924   |          0.00988778  |       0.0398873   |
| high_velocity_low_trust       |          0.211894   |          0.00722264  |       0.0330958   |
| foreign_x_location_mismatch   |          0.235935   |          0.0083987   |       0.0325077   |
| hour_sin                      |          0.0423031  |          0.00677374  |       0.0238883   |
| velocity_x_amount             |          0.060163   |          0.00203565  |       0.0227148   |
| log_amount                    |          0.00811326 |          0.00234301  |       0.0205637   |
| amount                        |          0.039288   |          0.00236379  |       0.019231    |
| amount_per_velocity           |          0.0315725  |          0.00371715  |       0.0188301   |
| cardholder_age                |          0.00971613 |          0           |       0.00971136  |
| device_trust_bin_medium       |          0.0614971  |          0.00378892  |       0.00391724  |
| merchant_category_Travel      |          0.00433463 |          0           |       0.00190516  |
| device_trust_bin_low          |          0.0532852  |          0           |       0.00148199  |
| merchant_category_Food        |          0.0129328  |          0           |       0.00123728  |
| age_group_middle              |          0.00322707 |          0           |       0.00122935  |
| age_group_senior              |          0.00333279 |          0.00113198  |       0.00115732  |
| merchant_category_Grocery     |          0.0316765  |          0           |       0.00095173  |
| merchant_category_Electronics |          0.0156445  |          0.000114169 |       0.000882951 |
| age_group_young               |          0.00418424 |          0           |       0.000656755 |

## 2. VIF Analysis

|    | feature                       |       VIF |
|---:|:------------------------------|----------:|
|  8 | risk_flags_count              | 233.697   |
| 21 | foreign_transaction           | 136.095   |
| 22 | location_mismatch             | 123.731   |
|  0 | amount                        |  20.3662  |
| 20 | device_trust_bin_very_low     |  19.7002  |
|  1 | device_trust_score            |  15.1064  |
| 16 | age_group_senior              |  12.7322  |
|  3 | cardholder_age                |  12.0872  |
| 18 | device_trust_bin_low          |  11.7773  |
| 10 | velocity_x_amount             |   9.50558 |
|  9 | amount_per_velocity           |   8.88025 |
| 19 | device_trust_bin_medium       |   4.00979 |
| 15 | age_group_middle              |   3.55228 |
| 25 | late_night_flag               |   3.0924  |
|  5 | log_amount                    |   2.86576 |
|  4 | transaction_hour              |   2.77811 |
|  6 | hour_sin                      |   2.55583 |
|  7 | hour_cos                      |   2.55477 |
|  2 | velocity_last_24h             |   2.16392 |
| 17 | age_group_young               |   2.07044 |
| 26 | late_night_x_low_trust        |   1.75891 |
| 12 | merchant_category_Food        |   1.60005 |
| 14 | merchant_category_Travel      |   1.591   |
| 13 | merchant_category_Grocery     |   1.58453 |
| 11 | merchant_category_Electronics |   1.57974 |
| 24 | high_velocity_low_trust       |   1.30691 |
| 23 | foreign_x_location_mismatch   |   1.18797 |

## 3. RFECV Results

- Optimal number of features: 23
- Selected features: amount, device_trust_score, velocity_last_24h, cardholder_age, transaction_hour, log_amount, hour_sin, hour_cos, risk_flags_count, amount_per_velocity, velocity_x_amount, merchant_category_Travel, age_group_middle, age_group_senior, device_trust_bin_low, device_trust_bin_medium, device_trust_bin_very_low, foreign_transaction, location_mismatch, foreign_x_location_mismatch, high_velocity_low_trust, late_night_flag, late_night_x_low_trust
