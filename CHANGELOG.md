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
  • population size  [ 800->1000 ]
  • generations  [ 200->300 ]
  • elitism  [ 0.075->0.07 ]
  • length reward  [ 0.4->0.5 ]
  • hidden layers  [ [256]->[256, 128] ]
