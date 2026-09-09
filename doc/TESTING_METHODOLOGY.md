# Test Design Methodology

## 1. Map the Space
List every feature, rule, move, score, boundary.

## 2. Extend and Fill the Space
For each variable/dimension:
- `< min` (rejected/clamped)
- `= min` (boundary)
- interior point
- `= max` (boundary)
- `> max` (rejected/clamped)

Verify opposites produce different results (CW ≠ CCW, L ≠ R).

## 3. Fill the Grid (all combos upfront)
- N=1: isolation
- N=2: pairwise
- N×N×N and beyond: all viable combinations
- Cross-reference all tests against 2+ independent engines before proceeding

## 4. Mutation Testing Loop
```
while kill_rate improves:
    add targeted tests (one per surviving mutant)
    run mutations → check kill rate
```

## 5. Triage Survivors
For each surviving mutant:
1. Is it dead code? (read only)
2. Is it reachable given the API?
3. Remove unreachable/dead mutations

## 6. Ship to Dev
- Ship failing tests
- Dev reads spec + test → fixes code
- If dev disputes test → ?
