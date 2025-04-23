## 0.0.0 (22-04-2025)
- Initiated archives
- Created run_config
- Bad model before refactoring

## 0.0.1 (23-04-2025)
- Test for new code refactoring

## 1.0.0 (23-04-2025)
- New hope! Changing selection method and selecting from whole population not only elites
- Variables changes:
  • selection method round-robin->tournament
  • [new] tournament size [ 5 ]
  • generations  [ 400->800 ]
  • crossover  [ 0.3->0.9 ]
  • mutation  [ 0.95->0.9 ]
  • mutation rate  [ 0.3->0.5 ]
  • elitism  [ 0.15->0.025 ]
  • suit reward  [ 1.5->1.0 ]
  • length reward  [ 0.2->0.3 ]
  • diversity reward  [ 0.3->0.4 ]
