
## Main study, synthetic intersection (train p = test p, random demand)

| method | reward | train p | test p | seeds | delay (s) | CV delay | non-CV delay | gap (s) | unfinished |
|---|---|---|---|---|---|---|---|---|---|
| dqn | ipw | 0.1 | 0.1 | 3 | 135.4 ± 10.5 | 60.5 | 143.5 | 83.1 ± 4.6 | 107 |
| dqn | naive | 0.1 | 0.1 | 3 | 185.1 ± 64.4 | 110.6 | 193.1 | 82.5 ± 14.4 | 142 |
| dqn | privileged | 0.1 | 0.1 | 3 | 138.8 ± 51.4 | 87.0 | 144.5 | 57.5 ± 14.7 | 107 |
| dqn | stratified | 0.1 | 0.1 | 3 | 141.3 ± 23.9 | 74.9 | 148.4 | 73.6 ± 23.7 | 113 |
| dqn_loopstate | pressure | 0.1 | 0.1 | 5 | 118.3 ± 5.2 | 113.7 | 118.8 | 5.1 ± 4.5 | 95 |
| fixed_time | - | - | 0.1 | 1 | 302.9 ± nan | 304.0 | 302.8 | -1.2 ± nan | 188 |
| fu2026_qm_atsc | - | 0.1 | 0.1 | 5 | 196.9 ± 14.6 | 187.3 | 197.9 | 10.6 ± 2.0 | 150 |
| max_pressure_cv | - | - | 0.1 | 1 | 143.4 ± nan | 75.0 | 150.7 | 75.7 ± nan | 106 |
| max_pressure_loop | - | - | 0.1 | 1 | 164.8 ± nan | 153.8 | 166.1 | 12.2 ± nan | 131 |
| sac_alpha0.2 | ipw | 0.1 | 0.1 | 3 | 142.9 ± 17.8 | 65.4 | 151.2 | 85.8 ± 7.9 | 109 |
| sac_alpha0.2 | naive | 0.1 | 0.1 | 3 | 156.5 ± 37.4 | 87.8 | 163.9 | 76.1 ± 28.2 | 121 |
| sac_alpha0.2 | privileged | 0.1 | 0.1 | 3 | 148.9 ± 24.7 | 72.1 | 157.1 | 85.1 ± 15.0 | 117 |
| sac_auto | ipw | 0.1 | 0.1 | 5 | 153.0 ± 6.7 | 78.0 | 161.1 | 83.1 ± 7.8 | 116 |
| sac_auto | loop_count | 0.1 | 0.1 | 5 | 154.4 ± 22.1 | 75.3 | 163.0 | 87.7 ± 6.2 | 114 |
| sac_auto | naive | 0.1 | 0.1 | 5 | 159.1 ± 16.0 | 87.5 | 166.9 | 79.4 ± 7.3 | 130 |
| sac_auto | privileged | 0.1 | 0.1 | 5 | 150.1 ± 9.7 | 71.4 | 158.6 | 87.2 ± 8.2 | 113 |
| sac_auto | stratified | 0.1 | 0.1 | 5 | 151.3 ± 12.0 | 75.5 | 159.5 | 84.1 ± 4.8 | 114 |
| sac_auto_loopstate | fused | 0.1 | 0.1 | 5 | 111.3 ± 5.8 | 104.1 | 112.0 | 7.9 ± 5.1 | 90 |
| sac_auto_loopstate | fused_cs | 0.1 | 0.1 | 5 | 114.8 ± 6.9 | 110.9 | 115.2 | 4.3 ± 2.7 | 92 |
| sac_auto_loopstate | ipw | 0.1 | 0.1 | 5 | 113.5 ± 10.4 | 93.0 | 115.7 | 22.8 ± 2.1 | 93 |
| sac_auto_loopstate | loop_count | 0.1 | 0.1 | 5 | 112.7 ± 6.5 | 107.7 | 113.2 | 5.4 ± 2.7 | 90 |
| sac_auto_loopstate | naive | 0.1 | 0.1 | 5 | 108.5 ± 8.6 | 89.8 | 110.5 | 20.7 ± 4.8 | 92 |
| sac_auto_loopstate | pressure | 0.1 | 0.1 | 5 | 127.4 ± 8.8 | 118.3 | 128.4 | 10.1 ± 6.9 | 97 |
| sac_auto_loopstate | privileged | 0.1 | 0.1 | 5 | 111.7 ± 8.6 | 107.2 | 112.1 | 4.9 ± 2.6 | 90 |
| sac_auto_loopstate | stratified | 0.1 | 0.1 | 5 | 115.1 ± 15.4 | 109.6 | 115.7 | 6.2 ± 3.5 | 93 |
| dqn | ipw | 0.2 | 0.2 | 3 | 100.7 ± 6.0 | 74.0 | 107.3 | 33.3 ± 10.3 | 93 |
| dqn | naive | 0.2 | 0.2 | 3 | 105.9 ± 33.0 | 81.6 | 111.9 | 30.3 ± 0.9 | 100 |
| dqn | privileged | 0.2 | 0.2 | 3 | 109.3 ± 26.6 | 92.6 | 113.4 | 20.7 ± 5.4 | 95 |
| dqn | stratified | 0.2 | 0.2 | 3 | 111.9 ± 41.8 | 91.2 | 117.0 | 25.9 ± 8.7 | 103 |
| dqn_loopstate | pressure | 0.2 | 0.2 | 5 | 123.2 ± 7.1 | 123.6 | 123.1 | -0.5 ± 8.9 | 95 |
| fixed_time | - | - | 0.2 | 1 | 302.9 ± nan | 305.1 | 302.3 | -2.8 ± nan | 188 |
| fu2026_qm_atsc | - | 0.2 | 0.2 | 5 | 189.6 ± 23.3 | 187.6 | 190.0 | 2.5 ± 3.8 | 148 |
| max_pressure_cv | - | - | 0.2 | 1 | 141.7 ± nan | 109.5 | 149.7 | 40.2 ± nan | 115 |
| max_pressure_loop | - | - | 0.2 | 1 | 164.8 ± nan | 161.6 | 165.7 | 4.0 ± nan | 131 |
| sac_alpha0.2 | ipw | 0.2 | 0.2 | 3 | 100.1 ± 22.2 | 74.0 | 106.5 | 32.5 ± 1.9 | 89 |
| sac_alpha0.2 | naive | 0.2 | 0.2 | 3 | 117.2 ± 29.5 | 93.1 | 123.1 | 30.0 ± 10.7 | 103 |
| sac_alpha0.2 | privileged | 0.2 | 0.2 | 3 | 104.0 ± 14.2 | 79.3 | 110.0 | 30.8 ± 5.3 | 93 |
| sac_auto | ipw | 0.2 | 0.2 | 5 | 115.6 ± 15.1 | 90.2 | 121.8 | 31.6 ± 3.7 | 100 |
| sac_auto | loop_count | 0.2 | 0.2 | 5 | 102.8 ± 6.5 | 80.6 | 108.3 | 27.7 ± 2.8 | 92 |
| sac_auto | naive | 0.2 | 0.2 | 5 | 128.2 ± 25.3 | 101.4 | 134.8 | 33.3 ± 4.4 | 105 |
| sac_auto | privileged | 0.2 | 0.2 | 5 | 111.3 ± 11.6 | 86.4 | 117.4 | 31.0 ± 3.0 | 98 |
| sac_auto | stratified | 0.2 | 0.2 | 5 | 111.6 ± 17.0 | 89.5 | 117.0 | 27.5 ± 2.8 | 98 |
| sac_auto_loopstate | fused | 0.2 | 0.2 | 5 | 111.9 ± 5.4 | 109.7 | 112.4 | 2.7 ± 4.6 | 88 |
| sac_auto_loopstate | fused_cs | 0.2 | 0.2 | 5 | 109.8 ± 2.5 | 107.2 | 110.5 | 3.3 ± 3.3 | 89 |
| sac_auto_loopstate | ipw | 0.2 | 0.2 | 5 | 105.1 ± 7.0 | 96.8 | 107.2 | 10.3 ± 5.1 | 87 |
| sac_auto_loopstate | loop_count | 0.2 | 0.2 | 5 | 109.9 ± 7.7 | 108.7 | 110.2 | 1.5 ± 3.9 | 87 |
| sac_auto_loopstate | naive | 0.2 | 0.2 | 5 | 107.5 ± 5.5 | 96.6 | 110.3 | 13.7 ± 2.5 | 87 |
| sac_auto_loopstate | pressure | 0.2 | 0.2 | 5 | 130.8 ± 11.2 | 128.9 | 131.3 | 2.4 ± 5.2 | 100 |
| sac_auto_loopstate | privileged | 0.2 | 0.2 | 5 | 106.6 ± 6.4 | 104.5 | 107.1 | 2.7 ± 2.3 | 87 |
| sac_auto_loopstate | stratified | 0.2 | 0.2 | 5 | 105.5 ± 6.6 | 103.1 | 106.1 | 3.0 ± 4.4 | 85 |
| sac_auto_loopstate_H0.3 | stratified | 0.2 | 0.2 | 3 | 100.9 ± 20.2 | 98.5 | 101.5 | 3.0 ± 2.6 | 86 |
| sac_auto_loopstate_H0.7 | stratified | 0.2 | 0.2 | 3 | 121.2 ± 12.3 | 119.7 | 121.6 | 1.8 ± 11.1 | 97 |
| sac_auto_loopstate_K100 | stratified | 0.2 | 0.2 | 3 | 109.0 ± 13.1 | 106.0 | 109.8 | 3.8 ± 3.4 | 92 |
| sac_auto_loopstate_K25 | stratified | 0.2 | 0.2 | 3 | 107.3 ± 16.6 | 107.0 | 107.4 | 0.4 ± 7.7 | 86 |
| dqn | ipw | 0.4 | 0.4 | 3 | 102.0 ± 52.4 | 93.5 | 107.7 | 14.2 ± 3.3 | 97 |
| dqn | naive | 0.4 | 0.4 | 3 | 92.3 ± 12.3 | 82.8 | 98.6 | 15.8 ± 3.5 | 88 |
| dqn | privileged | 0.4 | 0.4 | 3 | 101.5 ± 38.6 | 94.1 | 106.4 | 12.3 ± 5.7 | 88 |
| dqn | stratified | 0.4 | 0.4 | 3 | 106.7 ± 84.1 | 99.6 | 111.4 | 11.9 ± 2.1 | 93 |
| dqn_loopstate | pressure | 0.4 | 0.4 | 5 | 119.7 ± 5.5 | 121.7 | 118.4 | -3.2 ± 2.0 | 95 |
| fixed_time | - | - | 0.4 | 1 | 302.9 ± nan | 305.1 | 301.4 | -3.7 ± nan | 188 |
| fu2026_qm_atsc | - | 0.4 | 0.4 | 5 | 184.0 ± 15.6 | 182.6 | 184.9 | 2.2 ± 2.5 | 143 |
| max_pressure_cv | - | - | 0.4 | 1 | 151.5 ± nan | 139.4 | 159.4 | 20.0 ± nan | 121 |
| max_pressure_loop | - | - | 0.4 | 1 | 164.8 ± nan | 165.5 | 164.4 | -1.1 ± nan | 131 |
| sac_alpha0.2 | ipw | 0.4 | 0.4 | 3 | 88.4 ± 14.8 | 80.0 | 94.0 | 14.0 ± 5.1 | 83 |
| sac_alpha0.2 | naive | 0.4 | 0.4 | 3 | 90.9 ± 10.9 | 83.0 | 96.1 | 13.0 ± 1.7 | 85 |
| sac_alpha0.2 | privileged | 0.4 | 0.4 | 3 | 85.2 ± 3.8 | 78.4 | 89.8 | 11.4 ± 4.2 | 81 |
| sac_auto | ipw | 0.4 | 0.4 | 5 | 90.0 ± 5.5 | 81.6 | 95.5 | 14.0 ± 3.8 | 84 |
| sac_auto | loop_count | 0.4 | 0.4 | 5 | 92.0 ± 5.4 | 83.9 | 97.3 | 13.4 ± 2.4 | 84 |
| sac_auto | naive | 0.4 | 0.4 | 5 | 95.4 ± 4.9 | 87.8 | 100.3 | 12.5 ± 3.7 | 88 |
| sac_auto | privileged | 0.4 | 0.4 | 5 | 96.7 ± 19.4 | 89.2 | 101.7 | 12.6 ± 2.5 | 84 |
| sac_auto | stratified | 0.4 | 0.4 | 5 | 91.9 ± 7.4 | 84.2 | 96.9 | 12.7 ± 2.1 | 86 |
| sac_auto_loopstate | fused | 0.4 | 0.4 | 5 | 111.2 ± 6.9 | 112.5 | 110.4 | -2.1 ± 2.5 | 88 |
| sac_auto_loopstate | fused_cs | 0.4 | 0.4 | 5 | 109.0 ± 4.6 | 109.9 | 108.4 | -1.6 ± 3.5 | 87 |
| sac_auto_loopstate | ipw | 0.4 | 0.4 | 5 | 105.5 ± 7.4 | 103.8 | 106.6 | 2.8 ± 5.5 | 90 |
| sac_auto_loopstate | loop_count | 0.4 | 0.4 | 5 | 111.5 ± 11.4 | 111.4 | 111.6 | 0.2 ± 3.8 | 91 |
| sac_auto_loopstate | naive | 0.4 | 0.4 | 5 | 104.4 ± 9.0 | 103.2 | 105.2 | 1.9 ± 4.1 | 86 |
| sac_auto_loopstate | pressure | 0.4 | 0.4 | 5 | 126.7 ± 11.1 | 128.0 | 125.9 | -2.1 ± 3.1 | 96 |
| sac_auto_loopstate | privileged | 0.4 | 0.4 | 5 | 104.3 ± 5.3 | 105.7 | 103.4 | -2.3 ± 2.0 | 86 |
| sac_auto_loopstate | stratified | 0.4 | 0.4 | 5 | 128.8 ± 67.5 | 129.3 | 128.4 | -0.9 ± 2.6 | 97 |
| fixed_time | - | - | 1.0 | 1 | 302.9 ± nan | 302.9 | nan | nan ± nan | 188 |
| fu2026_qm_atsc | - | 1.0 | 1.0 | 5 | 177.8 ± 12.5 | 177.8 | nan | nan ± nan | 131 |
| max_pressure_cv | - | - | 1.0 | 1 | 164.8 ± nan | 164.8 | nan | nan ± nan | 131 |
| max_pressure_loop | - | - | 1.0 | 1 | 164.8 ± nan | 164.8 | nan | nan ± nan | 131 |
| sac_auto | loop_count | 1.0 | 1.0 | 5 | 90.3 ± 8.7 | 90.3 | nan | nan ± nan | 79 |
| sac_auto | privileged | 1.0 | 1.0 | 5 | 87.9 ± 5.9 | 87.9 | nan | nan ± nan | 77 |
| sac_auto_loopstate | privileged | 1.0 | 1.0 | 5 | 91.4 ± 5.1 | 91.4 | nan | nan ± nan | 77 |

## Cross-p transfer (SAC, auto temperature)

| method | reward | train p | test p | seeds | delay (s) | CV delay | non-CV delay | gap (s) | unfinished |
|---|---|---|---|---|---|---|---|---|---|
| sac_auto | ipw | 0.2 | 0.1 | 5 | 157.5 ± 14.3 | 77.7 | 166.2 | 88.4 ± 1.7 | 118 |
| sac_auto | ipw | 0.4 | 0.1 | 5 | 250.2 ± 41.4 | 162.4 | 259.6 | 97.2 ± 16.7 | 173 |
| sac_auto | loop_count | 0.2 | 0.1 | 5 | 139.6 ± 16.0 | 67.6 | 147.4 | 79.8 ± 11.3 | 108 |
| sac_auto | loop_count | 0.4 | 0.1 | 5 | 232.3 ± 99.9 | 135.5 | 242.7 | 107.2 ± 18.3 | 164 |
| sac_auto | loop_count | 1.0 | 0.1 | 5 | 733.5 ± 199.3 | 583.0 | 749.5 | 166.6 ± 10.8 | 296 |
| sac_auto | naive | 0.2 | 0.1 | 5 | 176.4 ± 11.6 | 94.1 | 185.2 | 91.2 ± 5.6 | 128 |
| sac_auto | naive | 0.4 | 0.1 | 5 | 264.7 ± 93.8 | 158.9 | 276.1 | 117.3 ± 18.2 | 179 |
| sac_auto | privileged | 0.2 | 0.1 | 5 | 154.4 ± 15.4 | 77.0 | 162.7 | 85.7 ± 5.6 | 114 |
| sac_auto | privileged | 0.4 | 0.1 | 5 | 207.1 ± 19.0 | 115.8 | 216.9 | 101.1 ± 20.4 | 152 |
| sac_auto | privileged | 1.0 | 0.1 | 5 | 572.2 ± 96.5 | 442.9 | 585.9 | 143.0 ± 27.2 | 279 |
| sac_auto | stratified | 0.2 | 0.1 | 5 | 154.6 ± 12.1 | 75.2 | 163.1 | 87.9 ± 5.0 | 114 |
| sac_auto | stratified | 0.4 | 0.1 | 5 | 207.4 ± 45.1 | 121.1 | 216.6 | 95.5 ± 20.5 | 142 |
| sac_auto | ipw | 0.1 | 0.2 | 5 | 152.3 ± 31.1 | 126.5 | 158.7 | 32.2 ± 5.4 | 122 |
| sac_auto | ipw | 0.4 | 0.2 | 5 | 103.4 ± 6.8 | 76.3 | 110.1 | 33.9 ± 3.9 | 93 |
| sac_auto | loop_count | 0.1 | 0.2 | 5 | 168.1 ± 52.4 | 143.6 | 174.2 | 30.6 ± 2.1 | 124 |
| sac_auto | loop_count | 0.4 | 0.2 | 5 | 102.3 ± 6.4 | 77.4 | 108.5 | 31.1 ± 4.0 | 91 |
| sac_auto | loop_count | 1.0 | 0.2 | 5 | 236.0 ± 83.4 | 188.1 | 247.7 | 59.6 ± 14.0 | 156 |
| sac_auto | naive | 0.1 | 0.2 | 5 | 171.9 ± 36.0 | 148.2 | 177.7 | 29.5 ± 4.4 | 133 |
| sac_auto | naive | 0.4 | 0.2 | 5 | 115.4 ± 12.5 | 88.0 | 122.2 | 34.1 ± 5.3 | 100 |
| sac_auto | privileged | 0.1 | 0.2 | 5 | 171.2 ± 24.6 | 149.7 | 176.5 | 26.8 ± 6.6 | 126 |
| sac_auto | privileged | 0.4 | 0.2 | 5 | 103.1 ± 5.3 | 77.4 | 109.4 | 31.9 ± 1.9 | 93 |
| sac_auto | privileged | 1.0 | 0.2 | 5 | 188.1 ± 35.2 | 145.6 | 198.6 | 53.0 ± 11.9 | 140 |
| sac_auto | stratified | 0.1 | 0.2 | 5 | 171.9 ± 39.4 | 147.2 | 178.0 | 30.7 ± 3.4 | 131 |
| sac_auto | stratified | 0.4 | 0.2 | 5 | 104.6 ± 4.5 | 79.0 | 110.9 | 31.9 ± 3.3 | 94 |
| sac_auto | ipw | 0.1 | 0.4 | 5 | 213.9 ± 88.0 | 204.7 | 220.0 | 15.2 ± 4.7 | 149 |
| sac_auto | ipw | 0.2 | 0.4 | 5 | 110.4 ± 3.5 | 101.1 | 116.6 | 15.6 ± 2.3 | 103 |
| sac_auto | loop_count | 0.1 | 0.4 | 5 | 235.4 ± 69.6 | 226.8 | 241.0 | 14.3 ± 10.1 | 154 |
| sac_auto | loop_count | 0.2 | 0.4 | 5 | 115.3 ± 15.8 | 109.0 | 119.5 | 10.5 ± 3.9 | 106 |
| sac_auto | loop_count | 1.0 | 0.4 | 5 | 113.5 ± 16.4 | 101.3 | 121.6 | 20.3 ± 2.2 | 100 |
| sac_auto | naive | 0.1 | 0.4 | 5 | 266.6 ± 42.8 | 259.9 | 271.3 | 11.4 ± 15.3 | 172 |
| sac_auto | naive | 0.2 | 0.4 | 5 | 128.4 ± 22.5 | 121.3 | 133.0 | 11.7 ± 3.7 | 108 |
| sac_auto | privileged | 0.1 | 0.4 | 5 | 262.4 ± 43.5 | 253.2 | 268.4 | 15.2 ± 5.8 | 173 |
| sac_auto | privileged | 0.2 | 0.4 | 5 | 130.5 ± 30.6 | 123.8 | 134.9 | 11.1 ± 5.2 | 108 |
| sac_auto | privileged | 1.0 | 0.4 | 5 | 111.3 ± 20.1 | 99.2 | 119.2 | 20.0 ± 2.2 | 95 |
| sac_auto | stratified | 0.1 | 0.4 | 5 | 279.5 ± 105.7 | 269.3 | 286.3 | 17.0 ± 7.4 | 180 |
| sac_auto | stratified | 0.2 | 0.4 | 5 | 128.3 ± 15.1 | 120.1 | 133.6 | 13.5 ± 2.5 | 111 |
| sac_auto | ipw | 0.1 | 1.0 | 5 | 264.6 ± 157.9 | 264.6 | nan | nan ± nan | 158 |
| sac_auto | ipw | 0.2 | 1.0 | 5 | 145.4 ± 24.7 | 145.4 | nan | nan ± nan | 116 |
| sac_auto | ipw | 0.4 | 1.0 | 5 | 94.8 ± 8.3 | 94.8 | nan | nan ± nan | 86 |
| sac_auto | loop_count | 0.1 | 1.0 | 5 | 374.6 ± 122.6 | 374.6 | nan | nan ± nan | 211 |
| sac_auto | loop_count | 0.2 | 1.0 | 5 | 193.7 ± 61.5 | 193.7 | nan | nan ± nan | 135 |
| sac_auto | loop_count | 0.4 | 1.0 | 5 | 108.2 ± 11.4 | 108.2 | nan | nan ± nan | 91 |
| sac_auto | naive | 0.1 | 1.0 | 5 | 319.7 ± 74.5 | 319.7 | nan | nan ± nan | 180 |
| sac_auto | naive | 0.2 | 1.0 | 5 | 154.8 ± 38.1 | 154.8 | nan | nan ± nan | 119 |
| sac_auto | naive | 0.4 | 1.0 | 5 | 97.5 ± 10.5 | 97.5 | nan | nan ± nan | 85 |
| sac_auto | privileged | 0.1 | 1.0 | 5 | 381.2 ± 101.3 | 381.2 | nan | nan ± nan | 206 |
| sac_auto | privileged | 0.2 | 1.0 | 5 | 224.7 ± 107.2 | 224.7 | nan | nan ± nan | 148 |
| sac_auto | privileged | 0.4 | 1.0 | 5 | 118.6 ± 29.8 | 118.6 | nan | nan ± nan | 97 |
| sac_auto | stratified | 0.1 | 1.0 | 5 | 532.0 ± 459.8 | 532.0 | nan | nan ± nan | 226 |
| sac_auto | stratified | 0.2 | 1.0 | 5 | 235.2 ± 144.0 | 235.2 | nan | nan ± nan | 152 |
| sac_auto | stratified | 0.4 | 1.0 | 5 | 104.5 ± 15.0 | 104.5 | nan | nan ± nan | 88 |

## Scenario: demand_fixed

| method | reward | train p | test p | seeds | delay (s) | CV delay | non-CV delay | gap (s) | unfinished |
|---|---|---|---|---|---|---|---|---|---|
| dqn | ipw | 0.1 | 0.1 | 3 | 152.2 ± 78.2 | 91.0 | 158.7 | 67.6 ± 6.1 | 118 |
| dqn | naive | 0.1 | 0.1 | 3 | 204.0 ± 108.7 | 150.3 | 209.6 | 59.4 ± 9.6 | 153 |
| dqn | privileged | 0.1 | 0.1 | 3 | 149.3 ± 26.7 | 101.6 | 154.3 | 52.7 ± 10.8 | 117 |
| dqn | stratified | 0.1 | 0.1 | 3 | 145.0 ± 37.8 | 92.6 | 150.5 | 57.9 ± 11.4 | 114 |
| fixed_time | - | - | 0.1 | 1 | 325.1 ± nan | 326.0 | 325.0 | -1.1 ± nan | 171 |
| fu2026_qm_atsc | - | 0.1 | 0.1 | 5 | 161.2 ± 7.1 | 161.7 | 161.1 | -0.6 ± 0.8 | 112 |
| max_pressure_cv | - | - | 0.1 | 1 | 154.1 ± nan | 95.3 | 160.4 | 65.0 ± nan | 112 |
| max_pressure_loop | - | - | 0.1 | 1 | 206.4 ± nan | 207.8 | 206.3 | -1.5 ± nan | 142 |
| sac_alpha0.2 | ipw | 0.1 | 0.1 | 3 | 137.1 ± 10.9 | 71.2 | 144.1 | 72.9 ± 3.2 | 109 |
| sac_alpha0.2 | naive | 0.1 | 0.1 | 3 | 157.3 ± 23.8 | 102.1 | 163.2 | 61.1 ± 27.4 | 118 |
| sac_alpha0.2 | privileged | 0.1 | 0.1 | 3 | 142.5 ± 29.4 | 82.3 | 148.9 | 66.6 ± 12.2 | 111 |
| sac_auto | ipw | 0.1 | 0.1 | 5 | 154.2 ± 14.4 | 91.2 | 160.9 | 69.7 ± 6.7 | 128 |
| sac_auto | loop_count | 0.1 | 0.1 | 5 | 141.6 ± 11.9 | 79.6 | 148.2 | 68.6 ± 2.6 | 113 |
| sac_auto | naive | 0.1 | 0.1 | 5 | 156.3 ± 10.1 | 92.1 | 163.1 | 71.1 ± 2.6 | 121 |
| sac_auto | privileged | 0.1 | 0.1 | 5 | 138.9 ± 4.5 | 76.9 | 145.5 | 68.6 ± 1.9 | 110 |
| sac_auto | stratified | 0.1 | 0.1 | 5 | 150.1 ± 17.6 | 88.2 | 156.7 | 68.5 ± 6.7 | 118 |
| sac_auto_loopstate | fused_cs | 0.1 | 0.1 | 5 | 115.9 ± 5.0 | 109.3 | 116.6 | 7.2 ± 4.1 | 83 |
| sac_auto_loopstate | ipw | 0.1 | 0.1 | 5 | 114.6 ± 6.5 | 98.4 | 116.4 | 17.9 ± 2.9 | 88 |
| sac_auto_loopstate | loop_count | 0.1 | 0.1 | 5 | 116.2 ± 3.3 | 112.3 | 116.6 | 4.3 ± 6.7 | 83 |
| sac_auto_loopstate | naive | 0.1 | 0.1 | 5 | 120.6 ± 15.8 | 104.6 | 122.3 | 17.6 ± 8.9 | 88 |
| sac_auto_loopstate | privileged | 0.1 | 0.1 | 5 | 116.2 ± 10.6 | 114.0 | 116.5 | 2.5 ± 4.3 | 83 |
| sac_auto_loopstate | stratified | 0.1 | 0.1 | 5 | 117.4 ± 6.3 | 112.5 | 118.0 | 5.5 ± 7.8 | 85 |
| dqn | ipw | 0.2 | 0.2 | 3 | 102.4 ± 29.1 | 78.4 | 108.4 | 30.0 ± 3.6 | 87 |
| dqn | naive | 0.2 | 0.2 | 3 | 99.7 ± 30.3 | 77.9 | 105.0 | 27.2 ± 5.2 | 89 |
| dqn | privileged | 0.2 | 0.2 | 3 | 108.9 ± 22.6 | 92.9 | 112.8 | 19.9 ± 11.4 | 94 |
| dqn | stratified | 0.2 | 0.2 | 3 | 102.7 ± 32.6 | 86.7 | 106.6 | 19.8 ± 4.7 | 84 |
| fixed_time | - | - | 0.2 | 1 | 325.1 ± nan | 330.6 | 323.8 | -6.8 ± nan | 171 |
| fu2026_qm_atsc | - | 0.2 | 0.2 | 5 | 165.4 ± 17.6 | 167.2 | 165.0 | -2.2 ± 0.9 | 113 |
| max_pressure_cv | - | - | 0.2 | 1 | 159.2 ± nan | 131.6 | 166.0 | 34.4 ± nan | 121 |
| max_pressure_loop | - | - | 0.2 | 1 | 206.4 ± nan | 209.0 | 205.8 | -3.2 ± nan | 142 |
| sac_alpha0.2 | ipw | 0.2 | 0.2 | 3 | 91.4 ± 16.9 | 71.3 | 96.3 | 25.0 ± 1.4 | 77 |
| sac_alpha0.2 | naive | 0.2 | 0.2 | 3 | 127.4 ± 41.7 | 106.5 | 132.5 | 26.1 ± 6.5 | 108 |
| sac_alpha0.2 | privileged | 0.2 | 0.2 | 3 | 93.4 ± 9.4 | 76.7 | 97.5 | 20.8 ± 5.7 | 80 |
| sac_auto | ipw | 0.2 | 0.2 | 5 | 96.0 ± 9.7 | 75.5 | 101.1 | 25.6 ± 1.5 | 82 |
| sac_auto | loop_count | 0.2 | 0.2 | 5 | 92.3 ± 10.1 | 74.4 | 96.8 | 22.4 ± 2.1 | 82 |
| sac_auto | naive | 0.2 | 0.2 | 5 | 116.6 ± 17.3 | 97.7 | 121.3 | 23.6 ± 2.9 | 95 |
| sac_auto | privileged | 0.2 | 0.2 | 5 | 94.5 ± 9.3 | 78.0 | 98.6 | 20.6 ± 3.5 | 83 |
| sac_auto | stratified | 0.2 | 0.2 | 5 | 97.0 ± 17.8 | 78.9 | 101.4 | 22.5 ± 1.8 | 80 |
| sac_auto_loopstate | fused_cs | 0.2 | 0.2 | 5 | 115.5 ± 9.7 | 113.0 | 116.2 | 3.2 ± 6.1 | 81 |
| sac_auto_loopstate | ipw | 0.2 | 0.2 | 5 | 112.2 ± 8.7 | 106.5 | 113.6 | 7.2 ± 2.5 | 83 |
| sac_auto_loopstate | loop_count | 0.2 | 0.2 | 5 | 116.6 ± 6.9 | 114.1 | 117.3 | 3.2 ± 4.5 | 85 |
| sac_auto_loopstate | naive | 0.2 | 0.2 | 5 | 113.9 ± 11.0 | 106.1 | 115.9 | 9.9 ± 2.4 | 84 |
| sac_auto_loopstate | privileged | 0.2 | 0.2 | 5 | 113.1 ± 6.7 | 110.0 | 114.0 | 4.0 ± 4.9 | 82 |
| sac_auto_loopstate | stratified | 0.2 | 0.2 | 5 | 111.3 ± 4.3 | 109.1 | 111.9 | 2.8 ± 3.8 | 82 |
| dqn | ipw | 0.4 | 0.4 | 3 | 86.2 ± 26.6 | 76.8 | 92.4 | 15.6 ± 3.3 | 72 |
| dqn | naive | 0.4 | 0.4 | 3 | 84.5 ± 11.4 | 74.8 | 90.9 | 16.1 ± 3.7 | 68 |
| dqn | privileged | 0.4 | 0.4 | 3 | 90.5 ± 36.7 | 84.5 | 94.5 | 10.1 ± 13.2 | 74 |
| dqn | stratified | 0.4 | 0.4 | 3 | 110.7 ± 146.0 | 105.4 | 114.2 | 8.8 ± 11.5 | 84 |
| fixed_time | - | - | 0.4 | 1 | 325.1 ± nan | 327.0 | 323.9 | -3.1 ± nan | 171 |
| fu2026_qm_atsc | - | 0.4 | 0.4 | 5 | 152.3 ± 0.8 | 154.7 | 150.8 | -3.9 ± 0.8 | 104 |
| max_pressure_cv | - | - | 0.4 | 1 | 184.4 ± nan | 171.0 | 193.2 | 22.2 ± nan | 129 |
| max_pressure_loop | - | - | 0.4 | 1 | 206.4 ± nan | 205.6 | 207.0 | 1.4 ± nan | 142 |
| sac_alpha0.2 | ipw | 0.4 | 0.4 | 3 | 77.8 ± 10.6 | 70.5 | 82.5 | 12.0 ± 1.3 | 66 |
| sac_alpha0.2 | naive | 0.4 | 0.4 | 3 | 86.9 ± 8.9 | 78.5 | 92.5 | 14.0 ± 0.5 | 73 |
| sac_alpha0.2 | privileged | 0.4 | 0.4 | 3 | 77.6 ± 5.3 | 70.9 | 82.1 | 11.2 ± 1.5 | 66 |
| sac_auto | ipw | 0.4 | 0.4 | 5 | 88.9 ± 8.0 | 80.3 | 94.5 | 14.2 ± 2.7 | 74 |
| sac_auto | loop_count | 0.4 | 0.4 | 5 | 88.3 ± 6.4 | 81.0 | 93.2 | 12.1 ± 1.8 | 71 |
| sac_auto | naive | 0.4 | 0.4 | 5 | 92.2 ± 7.7 | 84.0 | 97.7 | 13.7 ± 1.3 | 78 |
| sac_auto | privileged | 0.4 | 0.4 | 5 | 91.1 ± 10.2 | 83.9 | 96.0 | 12.1 ± 1.2 | 76 |
| sac_auto | stratified | 0.4 | 0.4 | 5 | 83.9 ± 4.3 | 77.0 | 88.6 | 11.6 ± 1.3 | 70 |
| sac_auto_loopstate | fused_cs | 0.4 | 0.4 | 5 | 111.5 ± 1.7 | 108.8 | 113.3 | 4.5 ± 1.7 | 80 |
| sac_auto_loopstate | ipw | 0.4 | 0.4 | 5 | 109.2 ± 9.2 | 105.5 | 111.8 | 6.3 ± 2.8 | 81 |
| sac_auto_loopstate | loop_count | 0.4 | 0.4 | 5 | 114.8 ± 5.9 | 112.5 | 116.4 | 3.9 ± 2.4 | 82 |
| sac_auto_loopstate | naive | 0.4 | 0.4 | 5 | 115.9 ± 11.3 | 112.1 | 118.5 | 6.4 ± 2.7 | 82 |
| sac_auto_loopstate | privileged | 0.4 | 0.4 | 5 | 113.4 ± 8.8 | 112.1 | 114.3 | 2.2 ± 0.7 | 82 |
| sac_auto_loopstate | stratified | 0.4 | 0.4 | 5 | 141.1 ± 81.1 | 138.4 | 143.0 | 4.6 ± 2.3 | 97 |
| fixed_time | - | - | 1.0 | 1 | 325.1 ± nan | 325.1 | nan | nan ± nan | 171 |
| fu2026_qm_atsc | - | 1.0 | 1.0 | 5 | 174.3 ± 29.4 | 174.3 | nan | nan ± nan | 121 |
| max_pressure_cv | - | - | 1.0 | 1 | 206.4 ± nan | 206.4 | nan | nan ± nan | 142 |
| max_pressure_loop | - | - | 1.0 | 1 | 206.4 ± nan | 206.4 | nan | nan ± nan | 142 |
| sac_auto | loop_count | 1.0 | 1.0 | 5 | 90.9 ± 8.8 | 90.9 | nan | nan ± nan | 74 |
| sac_auto | privileged | 1.0 | 1.0 | 5 | 98.6 ± 8.0 | 98.6 | nan | nan ± nan | 77 |
| sac_auto_loopstate | privileged | 1.0 | 1.0 | 5 | 100.7 ± 5.5 | 100.7 | nan | nan ± nan | 75 |

## Scenario: demand_peak

| method | reward | train p | test p | seeds | delay (s) | CV delay | non-CV delay | gap (s) | unfinished |
|---|---|---|---|---|---|---|---|---|---|
| dqn | ipw | 0.1 | 0.1 | 3 | 161.0 ± 24.2 | 90.7 | 168.4 | 77.7 ± 29.8 | 104 |
| dqn | naive | 0.1 | 0.1 | 3 | 236.7 ± 79.4 | 172.1 | 243.5 | 71.4 ± 22.1 | 153 |
| dqn | privileged | 0.1 | 0.1 | 3 | 190.4 ± 77.6 | 137.5 | 196.1 | 58.6 ± 2.7 | 125 |
| dqn | stratified | 0.1 | 0.1 | 3 | 167.5 ± 30.7 | 111.6 | 173.4 | 61.9 ± 22.6 | 111 |
| fixed_time | - | - | 0.1 | 1 | 311.6 ± nan | 311.7 | 311.6 | -0.1 ± nan | 165 |
| fu2026_qm_atsc | - | 0.1 | 0.1 | 5 | 171.6 ± 8.8 | 166.0 | 172.2 | 6.1 ± 1.4 | 102 |
| max_pressure_cv | - | - | 0.1 | 1 | 181.3 ± nan | 113.9 | 188.4 | 74.5 ± nan | 114 |
| max_pressure_loop | - | - | 0.1 | 1 | 228.3 ± nan | 221.2 | 229.1 | 7.8 ± nan | 147 |
| sac_alpha0.2 | ipw | 0.1 | 0.1 | 3 | 170.0 ± 8.6 | 90.0 | 178.5 | 88.5 ± 2.5 | 109 |
| sac_alpha0.2 | naive | 0.1 | 0.1 | 3 | 203.1 ± 20.8 | 134.2 | 210.3 | 76.1 ± 26.3 | 135 |
| sac_alpha0.2 | privileged | 0.1 | 0.1 | 3 | 188.1 ± 12.8 | 112.4 | 196.1 | 83.8 ± 12.4 | 128 |
| sac_auto | ipw | 0.1 | 0.1 | 5 | 201.8 ± 21.3 | 120.7 | 210.4 | 89.8 ± 12.6 | 132 |
| sac_auto | loop_count | 0.1 | 0.1 | 5 | 177.8 ± 7.4 | 102.8 | 185.8 | 82.9 ± 3.5 | 123 |
| sac_auto | naive | 0.1 | 0.1 | 5 | 199.9 ± 21.6 | 126.5 | 207.7 | 81.2 ± 1.9 | 137 |
| sac_auto | privileged | 0.1 | 0.1 | 5 | 182.7 ± 13.2 | 110.6 | 190.4 | 79.8 ± 4.6 | 127 |
| sac_auto | stratified | 0.1 | 0.1 | 5 | 194.8 ± 16.4 | 120.4 | 202.7 | 82.3 ± 10.3 | 134 |
| sac_auto_loopstate | fused_cs | 0.1 | 0.1 | 5 | 138.1 ± 9.8 | 137.1 | 138.2 | 1.1 ± 4.8 | 80 |
| sac_auto_loopstate | ipw | 0.1 | 0.1 | 5 | 136.1 ± 11.6 | 113.8 | 138.5 | 24.8 ± 3.6 | 86 |
| sac_auto_loopstate | loop_count | 0.1 | 0.1 | 5 | 133.6 ± 4.2 | 135.3 | 133.4 | -1.9 ± 4.9 | 76 |
| sac_auto_loopstate | naive | 0.1 | 0.1 | 5 | 145.4 ± 15.7 | 126.7 | 147.4 | 20.7 ± 6.8 | 92 |
| sac_auto_loopstate | privileged | 0.1 | 0.1 | 5 | 137.3 ± 15.2 | 136.0 | 137.5 | 1.5 ± 5.0 | 81 |
| sac_auto_loopstate | stratified | 0.1 | 0.1 | 5 | 130.4 ± 5.8 | 128.5 | 130.6 | 2.0 ± 5.2 | 75 |
| dqn | ipw | 0.2 | 0.2 | 3 | 139.8 ± 39.3 | 110.4 | 147.1 | 36.6 ± 9.4 | 91 |
| dqn | naive | 0.2 | 0.2 | 3 | 152.0 ± 55.4 | 123.8 | 159.0 | 35.2 ± 5.4 | 105 |
| dqn | privileged | 0.2 | 0.2 | 3 | 150.5 ± 18.2 | 132.7 | 154.8 | 22.1 ± 11.2 | 103 |
| dqn | stratified | 0.2 | 0.2 | 3 | 138.0 ± 22.1 | 115.8 | 143.5 | 27.7 ± 9.9 | 93 |
| fixed_time | - | - | 0.2 | 1 | 311.6 ± nan | 312.3 | 311.4 | -0.9 ± nan | 165 |
| fu2026_qm_atsc | - | 0.2 | 0.2 | 5 | 180.3 ± 24.0 | 174.7 | 181.8 | 7.1 ± 3.9 | 106 |
| max_pressure_cv | - | - | 0.2 | 1 | 188.8 ± nan | 148.8 | 198.7 | 49.9 ± nan | 125 |
| max_pressure_loop | - | - | 0.2 | 1 | 228.3 ± nan | 223.3 | 229.7 | 6.4 ± nan | 147 |
| sac_alpha0.2 | ipw | 0.2 | 0.2 | 3 | 136.2 ± 8.4 | 109.2 | 142.9 | 33.7 ± 5.3 | 97 |
| sac_alpha0.2 | naive | 0.2 | 0.2 | 3 | 168.5 ± 41.1 | 142.9 | 174.9 | 32.0 ± 16.7 | 118 |
| sac_alpha0.2 | privileged | 0.2 | 0.2 | 3 | 142.6 ± 29.0 | 118.2 | 148.6 | 30.5 ± 7.5 | 93 |
| sac_auto | ipw | 0.2 | 0.2 | 5 | 156.3 ± 13.4 | 127.7 | 163.4 | 35.7 ± 3.7 | 114 |
| sac_auto | loop_count | 0.2 | 0.2 | 5 | 156.5 ± 23.1 | 133.4 | 162.2 | 28.8 ± 5.3 | 110 |
| sac_auto | naive | 0.2 | 0.2 | 5 | 170.5 ± 4.1 | 141.9 | 177.6 | 35.7 ± 3.5 | 131 |
| sac_auto | privileged | 0.2 | 0.2 | 5 | 156.4 ± 19.7 | 131.6 | 162.5 | 30.9 ± 2.9 | 120 |
| sac_auto | stratified | 0.2 | 0.2 | 5 | 156.9 ± 15.2 | 131.4 | 163.2 | 31.8 ± 4.3 | 121 |
| sac_auto_loopstate | fused_cs | 0.2 | 0.2 | 5 | 134.2 ± 8.0 | 131.4 | 135.0 | 3.5 ± 2.9 | 83 |
| sac_auto_loopstate | ipw | 0.2 | 0.2 | 5 | 127.5 ± 10.9 | 117.1 | 130.1 | 13.1 ± 2.4 | 78 |
| sac_auto_loopstate | loop_count | 0.2 | 0.2 | 5 | 139.3 ± 9.3 | 137.5 | 139.9 | 2.4 ± 5.2 | 85 |
| sac_auto_loopstate | naive | 0.2 | 0.2 | 5 | 137.8 ± 12.6 | 127.1 | 140.6 | 13.5 ± 8.0 | 88 |
| sac_auto_loopstate | privileged | 0.2 | 0.2 | 5 | 132.8 ± 11.6 | 130.8 | 133.4 | 2.6 ± 3.6 | 80 |
| sac_auto_loopstate | stratified | 0.2 | 0.2 | 5 | 131.3 ± 10.9 | 126.7 | 132.6 | 5.8 ± 3.4 | 78 |
| dqn | ipw | 0.4 | 0.4 | 3 | 129.6 ± 34.6 | 118.2 | 137.2 | 18.9 ± 2.5 | 85 |
| dqn | naive | 0.4 | 0.4 | 3 | 130.9 ± 15.9 | 120.4 | 137.9 | 17.5 ± 4.6 | 87 |
| dqn | privileged | 0.4 | 0.4 | 3 | 140.4 ± 63.3 | 133.5 | 144.9 | 11.4 ± 16.8 | 96 |
| dqn | stratified | 0.4 | 0.4 | 3 | 140.8 ± 82.2 | 132.6 | 146.3 | 13.7 ± 11.0 | 89 |
| fixed_time | - | - | 0.4 | 1 | 311.6 ± nan | 312.2 | 311.2 | -1.1 ± nan | 165 |
| fu2026_qm_atsc | - | 0.4 | 0.4 | 5 | 178.4 ± 1.7 | 175.7 | 180.2 | 4.5 ± 0.5 | 107 |
| max_pressure_cv | - | - | 0.4 | 1 | 207.9 ± nan | 192.6 | 218.1 | 25.5 ± nan | 136 |
| max_pressure_loop | - | - | 0.4 | 1 | 228.3 ± nan | 225.8 | 230.1 | 4.3 ± nan | 147 |
| sac_alpha0.2 | ipw | 0.4 | 0.4 | 3 | 127.2 ± 29.4 | 117.1 | 133.8 | 16.7 ± 5.3 | 87 |
| sac_alpha0.2 | naive | 0.4 | 0.4 | 3 | 130.1 ± 11.0 | 118.8 | 137.5 | 18.7 ± 2.2 | 89 |
| sac_alpha0.2 | privileged | 0.4 | 0.4 | 3 | 119.0 ± 9.1 | 110.1 | 124.9 | 14.8 ± 3.0 | 77 |
| sac_auto | ipw | 0.4 | 0.4 | 5 | 135.3 ± 6.1 | 125.0 | 142.0 | 17.0 ± 1.3 | 93 |
| sac_auto | loop_count | 0.4 | 0.4 | 5 | 123.8 ± 5.4 | 115.7 | 129.1 | 13.4 ± 2.3 | 87 |
| sac_auto | naive | 0.4 | 0.4 | 5 | 137.9 ± 8.9 | 128.7 | 144.0 | 15.3 ± 1.2 | 99 |
| sac_auto | privileged | 0.4 | 0.4 | 5 | 137.3 ± 16.0 | 129.3 | 142.6 | 13.2 ± 3.1 | 93 |
| sac_auto | stratified | 0.4 | 0.4 | 5 | 127.8 ± 5.0 | 119.7 | 133.1 | 13.5 ± 2.4 | 86 |
| sac_auto_loopstate | fused_cs | 0.4 | 0.4 | 5 | 127.2 ± 7.8 | 123.8 | 129.5 | 5.7 ± 2.9 | 74 |
| sac_auto_loopstate | ipw | 0.4 | 0.4 | 5 | 132.1 ± 15.0 | 126.3 | 135.9 | 9.6 ± 2.7 | 81 |
| sac_auto_loopstate | loop_count | 0.4 | 0.4 | 5 | 130.3 ± 6.2 | 127.5 | 132.1 | 4.5 ± 3.1 | 73 |
| sac_auto_loopstate | naive | 0.4 | 0.4 | 5 | 133.2 ± 11.1 | 127.5 | 137.0 | 9.5 ± 2.5 | 83 |
| sac_auto_loopstate | privileged | 0.4 | 0.4 | 5 | 133.1 ± 8.1 | 129.7 | 135.3 | 5.5 ± 2.2 | 80 |
| sac_auto_loopstate | stratified | 0.4 | 0.4 | 5 | 148.3 ± 65.2 | 144.6 | 150.7 | 6.2 ± 4.0 | 85 |
| fixed_time | - | - | 1.0 | 1 | 311.6 ± nan | 311.6 | nan | nan ± nan | 165 |
| fu2026_qm_atsc | - | 1.0 | 1.0 | 5 | 198.4 ± 15.1 | 198.4 | nan | nan ± nan | 124 |
| max_pressure_cv | - | - | 1.0 | 1 | 228.3 ± nan | 228.3 | nan | nan ± nan | 147 |
| max_pressure_loop | - | - | 1.0 | 1 | 228.3 ± nan | 228.3 | nan | nan ± nan | 147 |
| sac_auto | loop_count | 1.0 | 1.0 | 5 | 120.2 ± 10.5 | 120.2 | nan | nan ± nan | 79 |
| sac_auto | privileged | 1.0 | 1.0 | 5 | 128.3 ± 5.6 | 128.3 | nan | nan ± nan | 81 |
| sac_auto_loopstate | privileged | 1.0 | 1.0 | 5 | 121.0 ± 8.2 | 121.0 | nan | nan ± nan | 74 |

## Scenario: fail1

| method | reward | train p | test p | seeds | delay (s) | CV delay | non-CV delay | gap (s) | unfinished |
|---|---|---|---|---|---|---|---|---|---|
| dqn_loopstate | pressure | 0.1 | 0.1 | 5 | 130.4 ± 8.1 | 128.5 | 130.6 | 2.1 ± 5.4 | 98 |
| max_pressure_loop | - | - | 0.1 | 1 | 176.1 ± nan | 173.2 | 176.3 | 3.1 ± nan | 132 |
| sac_auto_loopstate | fused_cs | 0.1 | 0.1 | 5 | 132.1 ± 15.2 | 131.5 | 132.2 | 0.7 ± 2.7 | 102 |
| sac_auto_loopstate | stratified | 0.1 | 0.1 | 5 | 131.7 ± 23.1 | 129.9 | 131.8 | 2.0 ± 2.8 | 101 |

## Scenario: noise0.05

| method | reward | train p | test p | seeds | delay (s) | CV delay | non-CV delay | gap (s) | unfinished |
|---|---|---|---|---|---|---|---|---|---|
| dqn_loopstate | pressure | 0.1 | 0.1 | 5 | 118.4 ± 8.2 | 113.8 | 118.9 | 5.0 ± 2.9 | 96 |
| max_pressure_loop | - | - | 0.1 | 1 | 165.3 ± nan | 155.0 | 166.5 | 11.5 ± nan | 131 |
| sac_auto_loopstate | fused_cs | 0.1 | 0.1 | 5 | 114.1 ± 6.6 | 110.0 | 114.5 | 4.5 ± 1.5 | 90 |
| sac_auto_loopstate | stratified | 0.1 | 0.1 | 5 | 107.8 ± 6.8 | 103.3 | 108.3 | 5.0 ± 3.9 | 88 |

## Scenario: noise0.1

| method | reward | train p | test p | seeds | delay (s) | CV delay | non-CV delay | gap (s) | unfinished |
|---|---|---|---|---|---|---|---|---|---|
| dqn_loopstate | pressure | 0.1 | 0.1 | 5 | 120.2 ± 7.9 | 116.8 | 120.6 | 3.8 ± 4.4 | 97 |
| max_pressure_loop | - | - | 0.1 | 1 | 164.3 ± nan | 153.4 | 165.5 | 12.2 ± nan | 131 |
| sac_auto_loopstate | fused_cs | 0.1 | 0.1 | 5 | 114.2 ± 5.6 | 109.8 | 114.7 | 4.9 ± 3.4 | 90 |
| sac_auto_loopstate | stratified | 0.1 | 0.1 | 5 | 107.2 ± 4.1 | 103.1 | 107.7 | 4.5 ± 4.2 | 88 |

## Scenario: noise0.2

| method | reward | train p | test p | seeds | delay (s) | CV delay | non-CV delay | gap (s) | unfinished |
|---|---|---|---|---|---|---|---|---|---|
| dqn_loopstate | pressure | 0.1 | 0.1 | 5 | 116.3 ± 9.6 | 113.3 | 116.6 | 3.3 ± 4.4 | 92 |
| max_pressure_loop | - | - | 0.1 | 1 | 166.0 ± nan | 155.2 | 167.2 | 12.1 ± nan | 131 |
| sac_auto_loopstate | fused_cs | 0.1 | 0.1 | 5 | 113.0 ± 5.1 | 109.3 | 113.4 | 4.1 ± 1.8 | 89 |
| sac_auto_loopstate | stratified | 0.1 | 0.1 | 5 | 108.0 ± 3.9 | 103.6 | 108.5 | 4.9 ± 5.5 | 87 |

## Scenario: uneven

| method | reward | train p | test p | seeds | delay (s) | CV delay | non-CV delay | gap (s) | unfinished |
|---|---|---|---|---|---|---|---|---|---|
| fixed_time | - | - | 0.1 | 1 | 302.9 ± nan | 312.7 | 301.5 | -11.2 ± nan | 188 |
| fu2026_qm_atsc | - | 0.1 | 0.1 | 5 | 199.9 ± 20.3 | 184.9 | 202.0 | 17.1 ± 9.5 | 154 |
| max_pressure_cv | - | - | 0.1 | 1 | 139.4 ± nan | 71.1 | 147.1 | 76.1 ± nan | 110 |
| max_pressure_loop | - | - | 0.1 | 1 | 164.8 ± nan | 154.4 | 166.1 | 11.7 ± nan | 131 |
| sac_auto | ipw | 0.1 | 0.1 | 5 | 186.5 ± 10.5 | 72.3 | 200.2 | 127.9 ± 11.8 | 135 |
| sac_auto | loop_count | 0.1 | 0.1 | 5 | 176.2 ± 13.8 | 65.9 | 189.4 | 123.5 ± 13.5 | 133 |
| sac_auto | naive | 0.1 | 0.1 | 5 | 176.1 ± 21.7 | 72.0 | 188.6 | 116.6 ± 21.8 | 135 |
| sac_auto | privileged | 0.1 | 0.1 | 5 | 183.3 ± 16.0 | 72.1 | 196.7 | 124.7 ± 13.2 | 137 |
| sac_auto | stratified | 0.1 | 0.1 | 5 | 177.4 ± 17.1 | 67.8 | 190.7 | 122.9 ± 17.1 | 134 |
| fixed_time | - | - | 0.2 | 1 | 302.9 ± nan | 310.6 | 300.3 | -10.3 ± nan | 188 |
| fu2026_qm_atsc | - | 0.2 | 0.2 | 5 | 194.0 ± 25.1 | 182.3 | 198.0 | 15.7 ± 16.0 | 150 |
| max_pressure_cv | - | - | 0.2 | 1 | 154.5 ± nan | 99.5 | 169.8 | 70.3 ± nan | 122 |
| max_pressure_loop | - | - | 0.2 | 1 | 164.8 ± nan | 162.7 | 166.0 | 3.3 ± nan | 131 |
| sac_auto | ipw | 0.2 | 0.2 | 5 | 121.4 ± 7.4 | 66.7 | 136.2 | 69.5 ± 6.2 | 96 |
| sac_auto | loop_count | 0.2 | 0.2 | 5 | 114.3 ± 13.8 | 66.0 | 127.3 | 61.3 ± 5.7 | 92 |
| sac_auto | naive | 0.2 | 0.2 | 5 | 124.6 ± 6.8 | 70.9 | 139.0 | 68.1 ± 7.1 | 104 |
| sac_auto | privileged | 0.2 | 0.2 | 5 | 117.8 ± 17.0 | 70.7 | 130.5 | 59.7 ± 9.7 | 101 |
| sac_auto | stratified | 0.2 | 0.2 | 5 | 118.2 ± 15.7 | 69.0 | 131.5 | 62.6 ± 8.8 | 95 |
| sac_auto_trained_uneven | ipw | 0.2 | 0.2 | 3 | 114.4 ± 16.8 | 64.3 | 128.2 | 63.8 ± 11.0 | 100 |
| sac_auto_trained_uneven | naive | 0.2 | 0.2 | 3 | 132.0 ± 40.1 | 78.8 | 146.8 | 68.1 ± 26.6 | 103 |
| sac_auto_trained_uneven | privileged | 0.2 | 0.2 | 3 | 119.4 ± 44.3 | 76.2 | 131.1 | 54.9 ± 9.8 | 96 |
| sac_auto_trained_uneven | stratified | 0.2 | 0.2 | 3 | 112.0 ± 24.2 | 64.8 | 124.9 | 60.1 ± 8.5 | 99 |
| fixed_time | - | - | 0.4 | 1 | 302.9 ± nan | 312.1 | 294.0 | -18.1 ± nan | 188 |
| fu2026_qm_atsc | - | 0.4 | 0.4 | 5 | 191.0 ± 22.5 | 176.0 | 204.6 | 28.5 ± 7.6 | 148 |
| max_pressure_cv | - | - | 0.4 | 1 | 169.7 ± nan | 122.9 | 207.2 | 84.3 ± nan | 128 |
| max_pressure_loop | - | - | 0.4 | 1 | 164.8 ± nan | 166.2 | 166.2 | 0.0 ± nan | 131 |
| sac_auto | ipw | 0.4 | 0.4 | 5 | 105.4 ± 11.8 | 74.5 | 129.6 | 55.0 ± 10.4 | 91 |
| sac_auto | loop_count | 0.4 | 0.4 | 5 | 104.6 ± 4.5 | 73.9 | 127.7 | 53.8 ± 10.8 | 93 |
| sac_auto | naive | 0.4 | 0.4 | 5 | 110.9 ± 8.9 | 74.5 | 139.1 | 64.6 ± 10.6 | 93 |
| sac_auto | privileged | 0.4 | 0.4 | 5 | 114.6 ± 13.5 | 82.1 | 140.0 | 57.9 ± 20.6 | 98 |
| sac_auto | stratified | 0.4 | 0.4 | 5 | 106.5 ± 10.6 | 77.8 | 128.4 | 50.7 ± 20.2 | 92 |

## Real intersection: cologne1

| method | reward | train p | test p | seeds | delay (s) | CV delay | non-CV delay | gap (s) | unfinished |
|---|---|---|---|---|---|---|---|---|---|
| dqn_loopstate | pressure | 0.1 | 0.1 | 5 | 21.2 ± 5.9 | 21.4 | 21.2 | -0.2 ± 1.1 | 18 |
| fixed_time | - | - | 0.1 | 1 | 127.2 ± nan | 128.8 | 127.0 | -1.8 ± nan | 64 |
| fu2026_qm_atsc | - | 0.1 | 0.1 | 5 | 143.8 ± 19.0 | 144.1 | 143.7 | -0.4 ± 2.1 | 58 |
| max_pressure_cv | - | - | 0.1 | 1 | 38.8 ± nan | 20.5 | 40.9 | 20.4 ± nan | 27 |
| max_pressure_loop | - | - | 0.1 | 1 | 20.9 ± nan | 21.0 | 20.9 | -0.1 ± nan | 16 |
| sac_auto | privileged | 0.1 | 0.1 | 3 | 36.5 ± 3.7 | 19.9 | 38.3 | 18.4 ± 2.1 | 24 |
| sac_auto | stratified | 0.1 | 0.1 | 3 | 36.2 ± 3.2 | 19.6 | 38.0 | 18.4 ± 0.2 | 23 |
| sac_auto_loopstate | fused | 0.1 | 0.1 | 5 | 25.8 ± 6.3 | 24.6 | 26.0 | 1.3 ± 1.4 | 23 |
| sac_auto_loopstate | fused_cs | 0.1 | 0.1 | 5 | 24.1 ± 1.6 | 22.8 | 24.3 | 1.5 ± 0.8 | 21 |
| sac_auto_loopstate | ipw | 0.1 | 0.1 | 5 | 47.2 ± 38.6 | 35.5 | 48.4 | 12.9 ± 5.9 | 32 |
| sac_auto_loopstate | loop_count | 0.1 | 0.1 | 5 | 43.8 ± 28.0 | 41.6 | 44.1 | 2.5 ± 4.2 | 27 |
| sac_auto_loopstate | naive | 0.1 | 0.1 | 5 | 53.4 ± 36.3 | 38.9 | 55.0 | 16.2 ± 8.3 | 31 |
| sac_auto_loopstate | pressure | 0.1 | 0.1 | 5 | 24.5 ± 2.2 | 23.8 | 24.6 | 0.9 ± 0.3 | 20 |
| sac_auto_loopstate | privileged | 0.1 | 0.1 | 5 | 24.2 ± 1.3 | 23.4 | 24.2 | 0.8 ± 0.2 | 21 |
| sac_auto_loopstate | stratified | 0.1 | 0.1 | 5 | 27.0 ± 2.3 | 23.1 | 27.4 | 4.3 ± 1.3 | 23 |
| dqn_loopstate | pressure | 0.2 | 0.2 | 5 | 18.6 ± 0.5 | 18.7 | 18.6 | -0.0 ± 0.4 | 15 |
| fixed_time | - | - | 0.2 | 1 | 127.2 ± nan | 127.6 | 127.1 | -0.5 ± nan | 64 |
| fu2026_qm_atsc | - | 0.2 | 0.2 | 5 | 133.9 ± 13.3 | 134.6 | 133.7 | -0.9 ± 2.1 | 57 |
| max_pressure_cv | - | - | 0.2 | 1 | 28.3 ± nan | 20.9 | 30.1 | 9.2 ± nan | 19 |
| max_pressure_loop | - | - | 0.2 | 1 | 20.9 ± nan | 21.1 | 20.9 | -0.2 ± nan | 16 |
| sac_auto | privileged | 0.2 | 0.2 | 3 | 68.1 ± 88.8 | 66.1 | 68.7 | 2.6 ± 26.3 | 33 |
| sac_auto | stratified | 0.2 | 0.2 | 3 | 26.0 ± 0.3 | 19.2 | 27.7 | 8.5 ± 0.7 | 20 |
| sac_auto_loopstate | fused | 0.2 | 0.2 | 5 | 30.2 ± 19.7 | 29.7 | 30.3 | 0.6 ± 0.3 | 23 |
| sac_auto_loopstate | fused_cs | 0.2 | 0.2 | 5 | 36.1 ± 36.7 | 35.5 | 36.3 | 0.8 ± 0.6 | 22 |
| sac_auto_loopstate | ipw | 0.2 | 0.2 | 5 | 73.6 ± 60.6 | 73.6 | 73.7 | 0.1 ± 15.3 | 34 |
| sac_auto_loopstate | loop_count | 0.2 | 0.2 | 5 | 31.8 ± 7.5 | 30.6 | 32.1 | 1.5 ± 1.4 | 26 |
| sac_auto_loopstate | naive | 0.2 | 0.2 | 5 | 45.1 ± 23.6 | 39.7 | 46.5 | 6.7 ± 2.5 | 32 |
| sac_auto_loopstate | pressure | 0.2 | 0.2 | 5 | 23.4 ± 1.7 | 23.1 | 23.4 | 0.3 ± 0.2 | 19 |
| sac_auto_loopstate | privileged | 0.2 | 0.2 | 5 | 40.6 ± 26.7 | 38.7 | 41.1 | 2.4 ± 3.0 | 26 |
| sac_auto_loopstate | stratified | 0.2 | 0.2 | 5 | 45.0 ± 35.8 | 45.6 | 44.8 | -0.8 ± 7.8 | 25 |
| dqn_loopstate | pressure | 0.4 | 0.4 | 5 | 33.7 ± 41.2 | 35.1 | 32.9 | -2.2 ± 6.8 | 19 |
| fixed_time | - | - | 0.4 | 1 | 127.2 ± nan | 128.1 | 126.6 | -1.5 ± nan | 64 |
| fu2026_qm_atsc | - | 0.4 | 0.4 | 5 | 130.8 ± 25.3 | 131.5 | 130.3 | -1.1 ± 1.0 | 59 |
| max_pressure_cv | - | - | 0.4 | 1 | 23.6 ± nan | 21.1 | 25.2 | 4.2 ± nan | 16 |
| max_pressure_loop | - | - | 0.4 | 1 | 20.9 ± nan | 21.0 | 20.9 | -0.1 ± nan | 16 |
| sac_auto | privileged | 0.4 | 0.4 | 3 | 28.4 ± 16.2 | 26.3 | 29.8 | 3.5 ± 1.9 | 23 |
| sac_auto | stratified | 0.4 | 0.4 | 3 | 24.4 ± 6.8 | 21.9 | 26.1 | 4.2 ± 0.8 | 20 |
| sac_auto_loopstate | fused | 0.4 | 0.4 | 5 | 23.8 ± 1.5 | 23.4 | 24.0 | 0.6 ± 0.3 | 20 |
| sac_auto_loopstate | fused_cs | 0.4 | 0.4 | 5 | 24.9 ± 1.6 | 24.6 | 25.1 | 0.5 ± 0.3 | 23 |
| sac_auto_loopstate | ipw | 0.4 | 0.4 | 5 | 43.6 ± 27.6 | 42.2 | 44.5 | 2.3 ± 3.2 | 28 |
| sac_auto_loopstate | loop_count | 0.4 | 0.4 | 5 | 54.7 ± 33.1 | 54.5 | 54.7 | 0.2 ± 4.7 | 29 |
| sac_auto_loopstate | naive | 0.4 | 0.4 | 5 | 38.8 ± 35.6 | 35.8 | 41.0 | 5.2 ± 4.6 | 24 |
| sac_auto_loopstate | pressure | 0.4 | 0.4 | 5 | 37.2 ± 35.5 | 36.1 | 38.0 | 1.9 ± 3.7 | 22 |
| sac_auto_loopstate | privileged | 0.4 | 0.4 | 5 | 32.8 ± 24.3 | 33.2 | 32.6 | -0.6 ± 3.0 | 22 |
| sac_auto_loopstate | stratified | 0.4 | 0.4 | 5 | 38.9 ± 43.0 | 40.0 | 38.3 | -1.6 ± 6.7 | 25 |
| fixed_time | - | - | 1.0 | 1 | 127.2 ± nan | 127.2 | nan | nan ± nan | 64 |
| fu2026_qm_atsc | - | 1.0 | 1.0 | 5 | 99.2 ± 7.2 | 99.2 | nan | nan ± nan | 55 |
| max_pressure_cv | - | - | 1.0 | 1 | 20.9 ± nan | 20.9 | nan | nan ± nan | 16 |
| max_pressure_loop | - | - | 1.0 | 1 | 20.9 ± nan | 20.9 | nan | nan ± nan | 16 |
| sac_auto_loopstate | privileged | 1.0 | 1.0 | 5 | 35.3 ± 29.1 | 35.3 | nan | nan ± nan | 23 |

## Real intersection: cologne3

| method | reward | train p | test p | seeds | delay (s) | CV delay | non-CV delay | gap (s) | unfinished |
|---|---|---|---|---|---|---|---|---|---|
| fixed_time | - | - | 0.1 | 1 | 83.1 ± nan | 82.7 | 83.1 | 0.4 ± nan | 57 |
| idqn_loopstate | pressure | 0.1 | 0.1 | 3 | 31.4 ± 32.4 | 32.7 | 31.3 | -1.4 ± 5.8 | 45 |
| isac_auto_loopstate | fused | 0.1 | 0.1 | 3 | 48.7 ± 18.0 | 47.0 | 49.0 | 1.9 ± 12.3 | 69 |
| isac_auto_loopstate | fused_cs | 0.1 | 0.1 | 3 | 34.0 ± 30.0 | 34.9 | 33.9 | -1.0 ± 5.4 | 52 |
| isac_auto_loopstate | ipw | 0.1 | 0.1 | 3 | 29.7 ± 5.1 | 25.1 | 30.3 | 5.1 ± 1.9 | 47 |
| isac_auto_loopstate | loop_count | 0.1 | 0.1 | 3 | 39.3 ± 31.8 | 39.2 | 39.3 | 0.1 ± 4.0 | 54 |
| isac_auto_loopstate | naive | 0.1 | 0.1 | 3 | 26.1 ± 5.5 | 21.2 | 26.7 | 5.5 ± 1.7 | 39 |
| isac_auto_loopstate | pressure | 0.1 | 0.1 | 3 | 61.6 ± 118.1 | 60.5 | 61.7 | 1.2 ± 3.7 | 74 |
| isac_auto_loopstate | privileged | 0.1 | 0.1 | 3 | 47.7 ± 6.5 | 48.2 | 47.6 | -0.6 ± 0.9 | 72 |
| isac_auto_loopstate | stratified | 0.1 | 0.1 | 3 | 31.7 ± 38.4 | 30.3 | 31.8 | 1.5 ± 1.1 | 47 |
| max_pressure_cv | - | - | 0.1 | 1 | 259.7 ± nan | 205.6 | 265.7 | 60.1 ± nan | 228 |
| max_pressure_loop | - | - | 0.1 | 1 | 44.0 ± nan | 45.4 | 43.9 | -1.5 ± nan | 35 |
| fixed_time | - | - | 0.2 | 1 | 83.1 ± nan | 83.0 | 83.1 | 0.1 ± nan | 57 |
| idqn_loopstate | pressure | 0.2 | 0.2 | 3 | 40.2 ± 44.6 | 40.8 | 40.1 | -0.8 ± 1.8 | 57 |
| isac_auto_loopstate | fused | 0.2 | 0.2 | 3 | 42.1 ± 72.3 | 42.1 | 42.1 | 0.0 ± 1.5 | 55 |
| isac_auto_loopstate | fused_cs | 0.2 | 0.2 | 3 | 36.3 ± 29.2 | 34.9 | 36.7 | 1.8 ± 1.9 | 50 |
| isac_auto_loopstate | ipw | 0.2 | 0.2 | 3 | 56.6 ± 95.5 | 55.2 | 56.9 | 1.6 ± 8.5 | 69 |
| isac_auto_loopstate | loop_count | 0.2 | 0.2 | 3 | 32.6 ± 30.6 | 33.5 | 32.4 | -1.1 ± 3.9 | 49 |
| isac_auto_loopstate | naive | 0.2 | 0.2 | 3 | 41.2 ± 63.8 | 39.8 | 41.5 | 1.7 ± 5.6 | 53 |
| isac_auto_loopstate | pressure | 0.2 | 0.2 | 3 | 37.2 ± 33.7 | 38.1 | 37.0 | -1.1 ± 4.4 | 53 |
| isac_auto_loopstate | privileged | 0.2 | 0.2 | 3 | 30.0 ± 28.6 | 28.5 | 30.4 | 1.9 ± 7.4 | 42 |
| isac_auto_loopstate | stratified | 0.2 | 0.2 | 3 | 43.0 ± 53.5 | 41.8 | 43.2 | 1.5 ± 7.8 | 58 |
| max_pressure_cv | - | - | 0.2 | 1 | 181.6 ± nan | 169.0 | 184.8 | 15.8 ± nan | 188 |
| max_pressure_loop | - | - | 0.2 | 1 | 44.0 ± nan | 44.7 | 43.9 | -0.8 ± nan | 35 |
| fixed_time | - | - | 1.0 | 1 | 83.1 ± nan | 83.1 | nan | nan ± nan | 57 |
| max_pressure_cv | - | - | 1.0 | 1 | 44.0 ± nan | 44.0 | nan | nan ± nan | 35 |
| max_pressure_loop | - | - | 1.0 | 1 | 44.0 ± nan | 44.0 | nan | nan ± nan | 35 |

## Real intersection: cologne8

| method | reward | train p | test p | seeds | delay (s) | CV delay | non-CV delay | gap (s) | unfinished |
|---|---|---|---|---|---|---|---|---|---|
| fixed_time | - | - | 0.1 | 1 | 122.8 ± nan | 123.7 | 122.7 | -1.0 ± nan | 69 |
| idqn_loopstate | pressure | 0.1 | 0.1 | 3 | 21.6 ± 2.4 | 20.8 | 21.7 | 0.9 ± 2.8 | 32 |
| isac_auto_loopstate | fused | 0.1 | 0.1 | 3 | 21.1 ± 1.0 | 18.4 | 21.4 | 3.0 ± 0.3 | 31 |
| isac_auto_loopstate | fused_cs | 0.1 | 0.1 | 3 | 21.2 ± 0.4 | 18.7 | 21.4 | 2.7 ± 1.1 | 31 |
| max_pressure_cv | - | - | 0.1 | 1 | 131.6 ± nan | 44.4 | 141.2 | 96.9 ± nan | 116 |
| max_pressure_loop | - | - | 0.1 | 1 | 23.0 ± nan | 23.3 | 22.9 | -0.3 ± nan | 32 |

## Real intersection: ingolstadt1

| method | reward | train p | test p | seeds | delay (s) | CV delay | non-CV delay | gap (s) | unfinished |
|---|---|---|---|---|---|---|---|---|---|
| dqn_loopstate | pressure | 0.1 | 0.1 | 5 | 16.5 ± 0.6 | 16.4 | 16.5 | 0.1 ± 0.3 | 16 |
| fixed_time | - | - | 0.1 | 1 | 28.4 ± nan | 26.9 | 28.5 | 1.6 ± nan | 21 |
| fu2026_qm_atsc | - | 0.1 | 0.1 | 5 | 35.2 ± 1.9 | 34.2 | 35.3 | 1.0 ± 0.6 | 20 |
| max_pressure_cv | - | - | 0.1 | 1 | 55.7 ± nan | 39.5 | 57.5 | 18.0 ± nan | 35 |
| max_pressure_loop | - | - | 0.1 | 1 | 15.7 ± nan | 15.2 | 15.7 | 0.6 ± nan | 14 |
| sac_auto | privileged | 0.1 | 0.1 | 3 | 42.9 ± 6.5 | 25.0 | 44.8 | 19.8 ± 2.4 | 28 |
| sac_auto | stratified | 0.1 | 0.1 | 3 | 35.5 ± 7.2 | 17.8 | 37.4 | 19.6 ± 8.8 | 24 |
| sac_auto_loopstate | fused | 0.1 | 0.1 | 5 | 18.3 ± 1.5 | 17.5 | 18.3 | 0.9 ± 0.3 | 19 |
| sac_auto_loopstate | fused_cs | 0.1 | 0.1 | 5 | 18.2 ± 0.7 | 17.4 | 18.3 | 0.9 ± 0.6 | 18 |
| sac_auto_loopstate | ipw | 0.1 | 0.1 | 5 | 20.9 ± 1.4 | 17.5 | 21.2 | 3.8 ± 0.6 | 22 |
| sac_auto_loopstate | loop_count | 0.1 | 0.1 | 5 | 18.6 ± 1.4 | 18.3 | 18.6 | 0.3 ± 0.7 | 19 |
| sac_auto_loopstate | naive | 0.1 | 0.1 | 5 | 21.2 ± 1.7 | 17.2 | 21.7 | 4.5 ± 0.5 | 23 |
| sac_auto_loopstate | pressure | 0.1 | 0.1 | 5 | 17.6 ± 0.4 | 17.2 | 17.6 | 0.5 ± 0.3 | 18 |
| sac_auto_loopstate | privileged | 0.1 | 0.1 | 5 | 18.6 ± 1.5 | 18.4 | 18.6 | 0.2 ± 0.4 | 19 |
| sac_auto_loopstate | stratified | 0.1 | 0.1 | 5 | 19.0 ± 1.1 | 17.9 | 19.1 | 1.2 ± 0.5 | 19 |
| dqn_loopstate | pressure | 0.2 | 0.2 | 5 | 17.1 ± 1.0 | 17.3 | 17.1 | -0.2 ± 0.4 | 16 |
| fixed_time | - | - | 0.2 | 1 | 28.4 ± nan | 28.2 | 28.4 | 0.2 ± nan | 21 |
| fu2026_qm_atsc | - | 0.2 | 0.2 | 5 | 34.5 ± 2.0 | 34.3 | 34.5 | 0.2 ± 0.8 | 21 |
| max_pressure_cv | - | - | 0.2 | 1 | 27.2 ± nan | 22.5 | 28.4 | 5.9 ± nan | 21 |
| max_pressure_loop | - | - | 0.2 | 1 | 15.7 ± nan | 15.7 | 15.7 | -0.1 ± nan | 14 |
| sac_auto | privileged | 0.2 | 0.2 | 3 | 21.0 ± 0.8 | 17.7 | 21.9 | 4.1 ± 0.7 | 17 |
| sac_auto | stratified | 0.2 | 0.2 | 3 | 20.0 ± 1.2 | 16.2 | 20.9 | 4.7 ± 1.3 | 16 |
| sac_auto_loopstate | fused | 0.2 | 0.2 | 5 | 18.0 ± 0.3 | 17.7 | 18.1 | 0.4 ± 0.1 | 17 |
| sac_auto_loopstate | fused_cs | 0.2 | 0.2 | 5 | 18.0 ± 0.5 | 17.8 | 18.1 | 0.3 ± 0.3 | 16 |
| sac_auto_loopstate | ipw | 0.2 | 0.2 | 5 | 18.6 ± 0.5 | 16.8 | 19.1 | 2.3 ± 0.3 | 17 |
| sac_auto_loopstate | loop_count | 0.2 | 0.2 | 5 | 18.0 ± 0.8 | 17.9 | 18.0 | 0.1 ± 0.2 | 18 |
| sac_auto_loopstate | naive | 0.2 | 0.2 | 5 | 18.8 ± 0.4 | 16.8 | 19.3 | 2.5 ± 0.4 | 17 |
| sac_auto_loopstate | pressure | 0.2 | 0.2 | 5 | 17.2 ± 0.5 | 17.1 | 17.2 | 0.1 ± 0.2 | 16 |
| sac_auto_loopstate | privileged | 0.2 | 0.2 | 5 | 17.9 ± 0.6 | 17.8 | 18.0 | 0.2 ± 0.3 | 18 |
| sac_auto_loopstate | stratified | 0.2 | 0.2 | 5 | 18.5 ± 0.7 | 17.9 | 18.6 | 0.7 ± 0.3 | 18 |
| dqn_loopstate | pressure | 0.4 | 0.4 | 5 | 16.5 ± 1.7 | 16.5 | 16.6 | 0.0 ± 0.3 | 15 |
| fixed_time | - | - | 0.4 | 1 | 28.4 ± nan | 28.3 | 28.4 | 0.1 ± nan | 21 |
| fu2026_qm_atsc | - | 0.4 | 0.4 | 5 | 36.0 ± 2.4 | 35.9 | 36.0 | 0.1 ± 0.4 | 25 |
| max_pressure_cv | - | - | 0.4 | 1 | 18.9 ± nan | 16.9 | 20.2 | 3.3 ± nan | 16 |
| max_pressure_loop | - | - | 0.4 | 1 | 15.7 ± nan | 15.6 | 15.7 | 0.1 ± nan | 14 |
| sac_auto | privileged | 0.4 | 0.4 | 3 | 17.5 ± 0.2 | 16.4 | 18.3 | 1.9 ± 0.4 | 16 |
| sac_auto | stratified | 0.4 | 0.4 | 3 | 17.6 ± 1.6 | 16.4 | 18.4 | 1.9 ± 0.2 | 16 |
| sac_auto_loopstate | fused | 0.4 | 0.4 | 5 | 17.4 ± 0.6 | 17.2 | 17.5 | 0.4 ± 0.1 | 15 |
| sac_auto_loopstate | fused_cs | 0.4 | 0.4 | 5 | 17.3 ± 0.9 | 17.0 | 17.5 | 0.5 ± 0.1 | 15 |
| sac_auto_loopstate | ipw | 0.4 | 0.4 | 5 | 18.4 ± 2.0 | 17.7 | 18.9 | 1.3 ± 0.3 | 16 |
| sac_auto_loopstate | loop_count | 0.4 | 0.4 | 5 | 18.0 ± 0.8 | 17.9 | 18.1 | 0.1 ± 0.2 | 16 |
| sac_auto_loopstate | naive | 0.4 | 0.4 | 5 | 18.0 ± 1.4 | 17.3 | 18.6 | 1.3 ± 0.2 | 16 |
| sac_auto_loopstate | pressure | 0.4 | 0.4 | 5 | 17.0 ± 0.7 | 16.9 | 17.1 | 0.3 ± 0.2 | 15 |
| sac_auto_loopstate | privileged | 0.4 | 0.4 | 5 | 18.0 ± 1.0 | 17.8 | 18.1 | 0.3 ± 0.1 | 15 |
| sac_auto_loopstate | stratified | 0.4 | 0.4 | 5 | 17.9 ± 1.3 | 17.7 | 18.1 | 0.4 ± 0.2 | 17 |
| fixed_time | - | - | 1.0 | 1 | 28.4 ± nan | 28.4 | nan | nan ± nan | 21 |
| fu2026_qm_atsc | - | 1.0 | 1.0 | 5 | 35.8 ± 1.7 | 35.8 | nan | nan ± nan | 29 |
| max_pressure_cv | - | - | 1.0 | 1 | 15.7 ± nan | 15.7 | nan | nan ± nan | 14 |
| max_pressure_loop | - | - | 1.0 | 1 | 15.7 ± nan | 15.7 | nan | nan ± nan | 14 |
| sac_auto_loopstate | privileged | 1.0 | 1.0 | 5 | 15.6 ± 0.6 | 15.6 | nan | nan ± nan | 13 |

## Real intersection: ingolstadt7

| method | reward | train p | test p | seeds | delay (s) | CV delay | non-CV delay | gap (s) | unfinished |
|---|---|---|---|---|---|---|---|---|---|
| fixed_time | - | - | 0.1 | 1 | 74.5 ± nan | 72.9 | 74.7 | 1.9 ± nan | 99 |
| idqn_loopstate | pressure | 0.1 | 0.1 | 3 | 39.7 ± 6.5 | 39.5 | 39.7 | 0.2 ± 10.0 | 91 |
| isac_auto_loopstate | fused | 0.1 | 0.1 | 3 | 39.1 ± 3.9 | 26.4 | 40.5 | 14.2 ± 2.2 | 75 |
| isac_auto_loopstate | fused_cs | 0.1 | 0.1 | 3 | 39.6 ± 1.5 | 26.9 | 41.1 | 14.2 ± 1.5 | 77 |
| isac_auto_loopstate | stratified | 0.1 | 0.1 | 3 | 40.8 ± 3.5 | 25.5 | 42.4 | 16.9 ± 3.9 | 81 |
| max_pressure_cv | - | - | 0.1 | 1 | 284.2 ± nan | 157.2 | 298.2 | 141.1 ± nan | 305 |
| max_pressure_loop | - | - | 0.1 | 1 | 33.4 ± nan | 32.6 | 33.4 | 0.8 ± nan | 76 |
