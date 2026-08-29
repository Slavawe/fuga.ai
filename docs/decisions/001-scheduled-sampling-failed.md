# ADR-001: Scheduled Sampling Failed

## Status: Accepted (negative result)

## Context
Exposure bias in recurrent state advance (h) caused decode quality
degradation when mix > 0. Scheduled Sampling (ε=0.15: blend predicted
byte into advance_h with probability ε) was proposed to close the gap.

## Decision
Scheduled Sampling does NOT help. ε=0.15 gave 4 bytes vs 17 bytes at ε=0.

## Rationale
The predicted argmax is almost always the dominant e/r bigram. Adding
this noise into h AMPLIFIES the attractor instead of teaching resilience.
In standard LLMs, SS works because draft errors are rare; here 15%
noise in every window smears W.

## Consequences
- Teacher-forcing loopback (train on generated output) is the correct path
- ε=0 remains optimal for current architecture
