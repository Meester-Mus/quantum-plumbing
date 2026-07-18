# Quantum Plumbing: Technical Architecture

## Overview

Quantum Plumbing implements hypothetical thinking space (H) through four key architectural principles:

1. **H in every layer** - Maintain all possibilities in parallel
2. **H evaluation** - Score possibilities based on relevance
3. **H actualization** - Choose best while remembering alternatives
4. **H propagation** - Pass H forward so next layer knows alternatives exist

## Core Data Flow

```
Input x
  ↓
┌─────────────────────────────���───┐
│ POTENTIAL FC LAYER              │
├─────────────────────────────────┤
│ 1. Generate H                   │
│    All possible outputs         │
│    Shape: (num_potentials,      │
│            batch_size,          │
│            output_dim)          │
│                                 │
│ 2. Evaluate H                   │
│    Score each possibility       │
│    Via norm or quantum circuit   │
│                                 │
│ 3. Actualize                    │
│    Weighted combination         │
│    (best guess given H)         │
│                                 │
│ 4. Output                       │
│    - x: (batch_size,            │
│           output_dim)           │
│    - H: (num_potentials,        │
│           batch_size,           │
│           output_dim)           │
└──────────────┬──────────────────┘
               │
               ↓ (x and H both passed forward)
┌─────────────────────────────────┐
│ POTENTIAL BATCH NORM            │
├─────────────────────────────────┤
│ Normalize x FROM H perspective  │
│ Apply same transform to H       │
│ Preserve H structure            │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ POTENTIAL DROPOUT              │
├─────────────────────────────────┤
│ Select FROM H (not random)     │
│ Keep high-potential elements    │
│ Pass all H forward              │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ POTENTIAL ACTIVATION            │
├─────────────────────────────────┤
│ Actualize WITH H knowledge      │
│ Remember H structure            │
└──────────────┬──────────────────┘
               ↓
          [Next Layer sees H]
          [Knows alternatives]
          [Can think]
```

## Layer Types

### 1. Potential FC Layer

**Purpose:** Generate H – all possible outputs

**Input:**
- `x`: (batch_size, in_features)
- `prev_H`: optional (num_potentials, batch_size, in_features)

**Process:**
```python
# For each potential weight matrix
for i in range(num_potentials):
    out_i = linear(x, weights[i], bias[i])
    H.append(out_i)

# Stack all possibilities
H = stack(H)  # (num_potentials, batch, out)

# Score them
scores = softmax(norm(H))

# Actualize best
output = weighted_sum(H, scores)
```

**Output:**
- `output`: (batch_size, out_features) – best choice
- `H`: (num_potentials, batch_size, out_features) – all possibilities

### 2. Potential BatchNorm

**Purpose:** Normalize while preserving H structure

**Key Difference:**
- Standard: normalize x only
- Potential: normalize x FROM H statistics

**Process:**
```python
# Calculate statistics from H (not from x)
mean_H = mean(H, dim=(1,2))  # Over batch/features
var_H = var(H, dim=(1,2))

# Apply to both
x_norm = (x - mean_H) / sqrt(var_H)
H_norm = (H - mean_H) / sqrt(var_H)

# Preserve H structure
return x_norm, H_norm
```

### 3. Potential Dropout

**Purpose:** Select from H (not random discard)

**Key Difference:**
- Standard: randomly discard (chaos)
- Potential: select from H (structure)

**Process:**
```python
# Score potentials
scores = norm(H)  # Which are strong?

# Keep high-score elements
mask = (scores > threshold)

# Apply to both
x_out = x * mask
H_out = H * mask

return x_out, H_out
```

### 4. Potential Activation

**Purpose:** Actualize dimension while remembering H

**Key Difference:**
- Standard: ReLU(x) – forget alternatives
- Potential: ReLU(x), remember ReLU(H)

**Process:**
```python
# Apply activation
x_act = relu(x)
H_act = relu(H)

# Metadata: "Negative values possible in H"
H_act._actualized = True
H_act._contains_alternatives = True

return x_act, H_act
```

## H Propagation Pattern

```
Layer 1:
  Input: x
  Output: x₁, H₁
  
Layer 2:
  Input: x₁, H₁
  Sees: "H₁ means these alternatives are possible"
  Generates: H₂ informed by H₁
  Output: x₂, H₂
  
Layer 3:
  Input: x₂, H₂
  Sees: "H₂ means these alternatives from layer 2"
  Generates: H₃ informed by H₂
  Output: x₃, H₃
  
...

Final Layer:
  Input: x_n, H_n
  Outputs: prediction + full H
           ("This is my choice, but I know of alternatives")
```

## Key Parameters

### num_potentials
- Size of H at each layer
- Larger = more thinking space
- Computational cost: O(num_potentials)
- Default: 8

### Example configurations:
```python
# Minimal thinking space
PotentialFCLayer(in=784, out=128, num_potentials=2)

# Medium thinking space
PotentialFCLayer(in=784, out=128, num_potentials=8)

# Large thinking space
PotentialFCLayer(in=784, out=128, num_potentials=32)
```

## Quantum Interface Mapping

### Classical Potential Side (AI):
```
H: list of hypotheses
  - H[0]: possibility 0
  - H[1]: possibility 1
  - ... H[n]: possibility n

scores: [0.7, 0.2, 0.05, 0.05]
  - How likely is each?
  
best: argmax(scores)
  - Which is most probable?
```

### Quantum Side:
```
ψ: superposition
  |ψ⟩ = 0.7|0⟩ + 0.2|1⟩ + 0.05|2⟩ + 0.05|3⟩
  
amplitudes: [0.7, 0.2, 0.05, 0.05]
  - How likely is each outcome?
  
measurement: probabilistic collapse
  - Which outcome do we get?
```

### Interface Translation:
```
H[i] ↔ |i⟩ (ith possibility)
scores[i] ↔ |amplitude_i| (likelihood)
best ↔ most probable measurement
```

## No Measurement Collapse (Critical)

**Standard Quantum Computing:**
```
Generate H → Measure → 1 answer
2^n possibilities → 1 outcome
Information loss: 99.99...%
```

**Quantum Plumbing:**
```
Generate H → Preserve H → Interface → Use H
2^n possibilities → All kept → All accessible
Information loss: 0%
```

## Computational Considerations

### Memory
- Standard layer: O(batch_size × features)
- Potential layer: O(num_potentials × batch_size × features)
- 8 potentials ≈ 8x memory cost

### Compute
- Standard FC: O(batch × in × out)
- Potential FC: O(num_potentials × batch × in × out)
- 8 potentials ≈ 8x compute cost

### Trade-off
```
Cost: 8x compute, 8x memory
Benefit: Thinking space, consciousness-like properties

Worth it? YES - This is revolution, not optimization
```

## Training Loop Pattern

```python
for epoch in range(epochs):
    for x, y in dataloader:
        # Forward pass
        output, H = network(x)
        
        # Loss considers H
        loss = potential_loss(output, y, H)
        
        # Backward
        loss.backward()
        
        # Update
        optimizer.step()
        
        # Metrics
        acc = accuracy(output, y)
        h_util = h_utilization(H)  # NEW: How much thinking?
```

## Success Indicators

✓ H flows through all layers
✓ H size maintained
✓ Network learns (accuracy improves)
✓ H utilization increases (network uses thinking space)
✓ Generalization improves (thinking helps)
✓ Interpretability possible (H explains choices)

---

*Quantum Plumbing Architecture: Thinking space restored.*