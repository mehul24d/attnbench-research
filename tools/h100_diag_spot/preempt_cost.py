"""A preempted Spot VM's cost from Compute Engine's own operation times (researcher, 16 Nov 2026:
"book every future preemption from Compute Engine's create and preemption times, not the cap").

    gcloud compute operations list --filter="targetLink~NAME" \
        --format='value(operationType,insertTime,endTime)' | python3 -I preempt_cost.py RATE

Billed from the insert operation's insertTime (create) to the preempted operation's endTime
(the VM stopped), rounded up to whole minutes, at RATE INR/h, rounded up to the rupee (the
teardown's convention). Prints "COST MINUTES CREATE END"; prints nothing when either time is
missing, and the driver then books the cap.
"""
import math
import sys
from datetime import datetime

rate = float(sys.argv[1])
create = end = None
for line in sys.stdin:
    f = line.rstrip("\n").split("\t")
    if len(f) < 3:
        continue
    if f[0] == "insert" and f[1]:
        create = datetime.fromisoformat(f[1])
    elif f[0] == "compute.instances.preempted" and f[2]:
        end = datetime.fromisoformat(f[2])
if create is None or end is None or end <= create:
    sys.exit(0)
minutes = math.ceil((end - create).total_seconds() / 60)
print(math.ceil(minutes * rate / 60), minutes, create.isoformat(), end.isoformat())
