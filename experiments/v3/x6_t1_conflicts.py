"""X6 step 4: the narrowest objective conflict - IDENTICAL decoded pixels (T1 only) with incompatible
labels, in a group touching the deployed training set (b84_train). Compared with X3's T1+T2 quarantine."""
import os, sys, collections
sys.argv = ["x"]
src = open("x3_conflicts.py").read().split("packed, packedm = np.packbits")[0]   # T1 unions only; stop before T2
exec(src)
groups = collections.defaultdict(list)
for i in range(n): groups[find(i)].append(i)
labels_of = collections.defaultdict(set); where_of = collections.defaultdict(set)
for _, r, lab, w in items: labels_of[ri[r]].add(lab); where_of[ri[r]].add(w)
cg = [m for m in groups.values() if len(set().union(*(labels_of[i] for i in m))) > 1 and any("b84_train" in where_of[i] for i in m)]
quar = sorted({p for p, r, lab, w in items if w == "b84_train" and any(ri[r] in m for m in cg)})
x3 = {l.strip() for l in open("x3_quarantine.txt") if l.strip()}
rel = {os.path.relpath(p, WT) for p in quar}
print(f"T1-only conflict groups touching b84_train: {len(cg)}; b84_train photos: {len(quar)}; all inside X3's 86: {rel <= x3}")
print("by class:", collections.Counter(p.split('/')[-2] for p in quar))
open("x6_t1_quarantine.txt", "w").write("".join(r + "\n" for r in sorted(rel)))
