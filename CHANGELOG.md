## 0.0.0 (22-04-2025)
- Initiated archives
- Created run_config
- Bad model before refactoring

## 0.0.1 (23-04-2025)
- Test for new code refactoring

## 1.0.0 (23-04-2025)
- New hope! Changing selection method and selecting from whole population not only elites
- Variables changes:
  • (~)selection method round-robin->tournament
  • [new] (~)tournament size [ 5 ]
  • (+)generations  [ 400->800 ]
  • (+)crossover  [ 0.3->0.9 ]
  • (+)mutation  [ 0.95->0.9 ]
  • (+)mutation rate  [ 0.3->0.5 ]
  • (+)elitism  [ 0.15->0.025 ]
  • (-)suit reward  [ 1.5->1.0 ]
  • (+)length reward  [ 0.2->0.3 ]
  • (+)diversity reward  [ 0.3->0.4 ]

## 1.1.0 (24-04-2025)
- Softening mutation by a lot to not have random weights
- Higher elitism for better preservation
- Lower tournament size, so it won't converge to early
- Variables changes:
  • (-)tournament size [ 5->3 ]
  • (-)crossover  [ 0.9->0.7 ]
  • (-)mutation  [ 0.9->0.3 ]
  • (-)mutation rate  [ 0.5->0.1 ]
  • (-)perturbation scale  [ 0.8->0.2 ]
  • (+)elitism  [ 0.025->0.075 ]
  • (+)suit reward  [ 1.0->1.2 ]
  • (+)length reward  [ 0.3->0.4 ]

## 1.1.1 (24-04-2025)
- Getting more complexity by more generations and more hidden layers
- Fixed best color reward to give based on best contract in the color not for avg
- Variables changes:
  • (+)population size  [ 200->300 ]
  • (+)generations  [ 800->1000 ]
  • (-)elitism  [ 0.075->0.07 ]
  • (+)length reward  [ 0.4->0.5 ]
  • (+)hidden layers  [ [256]->[256, 128] ]

## 1.2.0 (24-04-2025)
- Adding colors lengths to input so the model learns to bid based on hand shape
- Changing imps linear difference to[(imps difference) ^ 1.5] punish crazy contracts and reward very good ones
- Variables changes:
  • (+)input size  [ 90->94 ]

## 1.2.1 (24-04-2025)
- Removing hand one hot encoding but adding points to see the difference
- Variables changes:
  • (+)generations [ 1000->1200 ]
  • (+)input size  [ 94->42 ]
  • (-)hidden layers  [ [256, 128]->[256] ]
  • (-)last batch  [ 800->-1 ]

## 1.2.2 (26-04-2025)
- Removing suit reward
- Variables changes:
  • (-)generations  [ 1200->1000 ]
  • (+)[new] batches per generation  [ 1->2 ]
  • (-)[new] imps to power  [ 1.5->1 ]
  • (-)reward for best suit  [ 1.2->0 ]
  • (+)bidding length bonus  [ 0.5->0.4 ]
  • (-)diversity bonus  [ 0.4->0.3 ]

## 1.2.3 (28-04-2025)
- Changing scoring and going for more diversity
- Variables changes:
  • (+)mutation probability  [ 0.3->0.4 ]
  • (+)mutation rate  [ 0.1->0.15 ]
  • (+)perturbation scale  [ 0.2->0.3 ]
  • (-)elitism  [ 0.07->0.03 ]
  • (+)suit reward  [ 0->0.1 ]
  • (-)bidding length bonus  [ 0.4->0.15 ]
  • (+)diversity bonus  [ 0.3->0 ]

## 1.2.4 (29-04-2025)
- Adding phases:
  • early: big research -> ramping population
  • middle: main phase -> islands (RL in future)
  • late: perfecting (MAP-Elites in future)
- Added log that summarize every n generations
- Variables changes:
  • (~)[new] phases
  • (+)population size  [ 300->600 ]
  • (+)mutation rate  [ 0.1->0.2 ]
  • (~)hidden layers  [ [256]->[128, 64] ]

## 1.2.5 (29-04-2025)
- fixed log and islands
- Variables changes:
  • (~)[new]better than pass bonus  [ 0.8 ]
  • (+)good suit reward  [ 0.1->0.2 ]

## 1.2.6 (29-04-2025)
- fixed a big bug with hand index
