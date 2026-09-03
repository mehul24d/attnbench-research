#!/usr/bin/env bash
# Read-only GCP status check for attnbench GPU work.
# Re-runs the Step 2 diagnostics: account/billing/API state, quota, accelerator
# availability, and existing resources. Makes no changes to the project.
set -uo pipefail

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGIONS=("asia-south1" "us-central1")
ZONES=("asia-south1-a" "asia-south1-b" "asia-south1-c")

if [[ -z "$PROJECT" ]]; then
  echo "No project set. Run: gcloud config set project PROJECT_ID" >&2
  exit 1
fi

hr() { printf '%s\n' "----------------------------------------------------------------"; }

echo "GCP Status — project: $PROJECT"
hr

echo "## Account / Billing"
gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>&1 | sed 's/^/Active account: /'
BILLING_JSON=$(gcloud billing projects describe "$PROJECT" --format=json 2>&1)
if echo "$BILLING_JSON" | grep -q '"billingEnabled"'; then
  echo "$BILLING_JSON" | python3 -c "
import json,sys
d = json.load(sys.stdin)
print(f\"Billing account: {d.get('billingAccountName','?')}\")
print(f\"Billing enabled: {d.get('billingEnabled', False)}\")
"
else
  echo "Could not read billing info (no access, or billing not linked):"
  echo "$BILLING_JSON"
fi
hr

echo "## Compute Engine API"
if gcloud services list --enabled --project="$PROJECT" 2>/dev/null | grep -q compute.googleapis.com; then
  echo "compute.googleapis.com: ENABLED"
else
  echo "compute.googleapis.com: NOT enabled"
  echo "  -> Enable with: gcloud services enable compute.googleapis.com --project=$PROJECT"
  echo "  (Everything below requires this API; skipping remaining checks.)"
  exit 0
fi
hr

echo "## Global quota (ALL_REGIONS)"
gcloud compute project-info describe --project="$PROJECT" --format=json 2>/dev/null | python3 -c "
import json,sys
d = json.load(sys.stdin)
for q in d.get('quotas', []):
    if q['metric'] in ('CPUS_ALL_REGIONS', 'GPUS_ALL_REGIONS'):
        print(f\"{q['metric']:<20} limit={q['limit']:<10} usage={q['usage']}\")
"
hr

echo "## Regional quota (CPU + all GPU-family lines)"
for r in "${REGIONS[@]}"; do
  echo "--- $r ---"
  gcloud compute regions describe "$r" --project="$PROJECT" --format=json 2>/dev/null | python3 -c "
import json,sys
d = json.load(sys.stdin)
for q in d.get('quotas', []):
    m = q['metric']
    if m == 'CPUS' or 'GPU' in m or m in ('INSTANCES','A2_CPUS'):
        if q['limit'] or q['usage']:
            print(f\"  {m:<38} limit={q['limit']:<10} usage={q['usage']}\")
"
done
hr

echo "## Accelerator types physically offered (asia-south1 zones)"
for z in "${ZONES[@]}"; do
  echo "--- $z ---"
  gcloud compute accelerator-types list --project="$PROJECT" --filter="zone:$z" --format="value(name)" 2>/dev/null | sed 's/^/  /'
done
hr

echo "## Existing resources (things that could be costing money)"
echo "Instances:"
gcloud compute instances list --project="$PROJECT" --format="table(name,zone,machineType.basename(),status)" 2>/dev/null
echo "Disks:"
gcloud compute disks list --project="$PROJECT" --format="table(name,zone,sizeGb,status)" 2>/dev/null
echo "Reserved addresses:"
gcloud compute addresses list --project="$PROJECT" --format="table(name,region,address,status)" 2>/dev/null
hr
echo "Done."
