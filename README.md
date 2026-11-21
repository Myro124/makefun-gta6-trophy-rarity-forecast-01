# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Myro124/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Myro124/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one: for every PlayStation 5 trophy, five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate on Exophase at the case's target, scored by pinball loss on the logit scale.

The model in `main.py` is an earners-over-tracked decomposition. A trophy's rate is earners divided by players tracked, and the two grow at different speeds: tracked players jump when Exophase refreshes its page (launch night it sat at 638 players for hours, then jumped to 3,868 and every rate but the one at 100% fell), while earners grow as players actually play. So the program fits, per trophy, how earners grow with log time since launch, fits how tracked players grow the same way, and divides the two forecasts. It also fits the rate's own log-time trend, averages the two in logit space, and shrinks toward the last observed rate when the fit is noisy, with each trophy's slope borrowing strength from its tier. When the whole history is flat, a refresh is overdue: the median is diluted toward the rate the refresh would leave, with a wide spread. A trophy at 100% stays at 100%.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory.

---

<sub>Entered on [make.fun](https://make.fun/?ref=gyq1s73s&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
