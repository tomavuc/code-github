#!/usr/bin/env python3
import math, csv, sys, re
from collections import Counter

# ───── CONFIG ────────────────────────────────────────────────────────────────
ALIGN_FILE      = "ispG.aln-clustalw"   # your MUSCLE CLUSTAL file
ECOLI_ID_SUBSTR = "AAC75568.1"            # substring to find E. coli row
WINDOW_OFFSET   = 50                    # expand ± this many columns around motif
TOP_N_ENTROPY   = 15                    # how many top-entropy cols to penalize
# ──────────────────────────────────────────────────────────────────────────────

# 1) load CLUSTAL alignment (manual parse, no Biopython needed)
sequences = {}
with open(ALIGN_FILE) as f:
    for L in f:
        L = L.rstrip("\n")
        if not L.strip() or L.startswith("CLUSTAL") or L[0].isspace():
            continue
        parts = L.split()
        if len(parts) < 2:
            continue
        seq_id, chunk = parts[0], parts[1]
        sequences.setdefault(seq_id, []).append(chunk)
for k in sequences:
    sequences[k] = "".join(sequences[k])

# 2) find the E. coli sequence
eco_keys = [k for k in sequences if ECOLI_ID_SUBSTR in k]
if not eco_keys:
    sys.exit(f"❌ Cannot find any ID containing '{ECOLI_ID_SUBSTR}' in alignment headers.")
ecoli_id = eco_keys[0]
ecoli_seq = sequences[ecoli_id]
L = len(ecoli_seq)

# 3) compute per‐column Shannon entropy (ignore gaps)
entropy = []
for col in zip(*sequences.values()):
    freq  = Counter(a for a in col if a != "-")
    n     = sum(freq.values())
    if n == 0:
        entropy.append(0.0)
    else:
        entropy.append(-sum((v/n) * math.log2(v/n) for v in freq.values()))

# 4) locate candidate motif columns:
#    - pick the 3 Cys columns in E. coli with *lowest* entropy
#    - pick the 1  Glu column in E. coli with *lowest* entropy
cys_cols = [i for i,ch in enumerate(ecoli_seq) if ch == "C"]
glu_cols = [i for i,ch in enumerate(ecoli_seq) if ch == "E"]
if len(cys_cols) < 3 or len(glu_cols) < 2:
    sys.exit("❌ Not enough Cys/E in the E. coli sequence to find motifs.")

# sort them by conservation (entropy ascending)
cys_cols_sorted = sorted(cys_cols, key=lambda i: entropy[i])[:3]
glu_col        = sorted(glu_cols, key=lambda i: entropy[i])[1]
motif_cols     = sorted(cys_cols_sorted + [glu_col])

print(f"🔎 Detected motif columns at alignment indices: {motif_cols}")
print("   (residues at those positions in E. coli → " +
      ", ".join(f"{i}:{ecoli_seq[i]}" for i in motif_cols) + ")")

# 5) define the binding‐site window
win_start = max(0, motif_cols[0] - WINDOW_OFFSET)
win_end   = min(L-1, motif_cols[-1] + WINDOW_OFFSET)
binding_cols = list(range(win_start, win_end + 1))
print(f"📐 Binding-site window: columns {win_start}–{win_end} (length {len(binding_cols)})")

# 6) pick top-entropy columns *outside* that window for penalty_topN
nonbind = [i for i in range(L) if i not in binding_cols]
top_entropy_cols = sorted(nonbind, key=lambda i: entropy[i], reverse=True)[:TOP_N_ENTROPY]

# 7) build & write feature table
with open("ispG_MSA_features.csv", "w", newline="") as out:
    w = csv.writer(out)
    w.writerow([
        "id",
        "pid_total",
        "mean_entropy_bind",
        "median_entropy_bind",
        "penalty_bind",
        f"penalty_top{TOP_N_ENTROPY}",
        "gaps_bind"
    ])
    for sid, seq in sequences.items():
        # a) overall PID
        match = sum(1 for a,b in zip(seq, ecoli_seq) if a == b and a != "-")
        pid_total = match / L

        # b) entropy stats over binding window
        bind_vals = [entropy[i] for i in binding_cols]
        mean_bind = sum(bind_vals) / len(bind_vals)
        med_bind  = sorted(bind_vals)[len(bind_vals)//2]

        # c) penalties where they *differ* from E. coli
        pen_bind = sum(entropy[i] for i in binding_cols if seq[i] != ecoli_seq[i])
        pen_topN = sum(entropy[i] for i in top_entropy_cols if seq[i] != ecoli_seq[i])

        # d) gaps in the window
        gaps_bind = sum(1 for i in binding_cols if seq[i] == "-")

        w.writerow([
            sid,
            f"{pid_total:.4f}",
            f"{mean_bind:.4f}",
            f"{med_bind:.4f}",
            f"{pen_bind:.4f}",
            f"{pen_topN:.4f}",
            gaps_bind
        ])

print("✅ ispG_MSA_features.csv generated with", len(sequences), "rows.")
