# The machine image is locked to the family it was captured from

**Status:** blocking the A100 arm. The fix needs a G2 instance up, so it is
scheduled for the next session that has one anyway; it costs minutes on
hardware already being paid for.

## What happened

On 2026-09-05, four attempts to boot `a2-ultragpu-1g` from
`attnbench-l4-image-v4-20260903` failed, each on a different inherited
property, and none of them on capacity. Nothing was created and nothing was
billed by these attempts, but they consumed most of a session.

| # | target | rejection |
|---|---|---|
| 1 | `a2-ultragpu-1g` | `Invalid value for field 'resource.disks[0].interface': 'NVME'` |
| 2 | `a2-ultragpu-1g` + `--boot-disk-interface=SCSI` | **identical** — the flag is ignored |
| 3 | `c3-standard-4` | `[c3-standard-4, nvidia-l4] features are not compatible` |
| 4 | `c3-standard-4` + `--accelerator=count=0` | `accelerator type must be specified` |

## Why it cannot be worked around at create time

A machine image records the full instance shape. Three properties matter:

| property | v4 value | overridable at create? |
|---|---|---|
| `machineType` | `g2-standard-8` | **yes** — `--machine-type` |
| `guestAccelerators` | `nvidia-l4` | **replaceable only** — never clearable |
| `disks[0].interface` | `NVME` | **no** |

`--boot-disk-interface` shapes a boot disk that *gcloud creates*. When the disk
comes from a machine image, the interface is part of the saved disk record and
the flag is ignored — attempt 2 produced a byte-identical error to attempt 1.

`--accelerator` requires `type=`; gcloud rejects `count=0` alone, and there is
no `--no-accelerator`. So an inherited accelerator can be swapped but not
removed.

Those two combine into a closed door:

- Keeping the L4 means only `g2` will accept the image — the contended pool.
- Escaping `g2` means clearing the accelerator — which cannot be done.

**So no CPU family can be used as a cheap seed**, which was the natural
workaround and is why it is worth writing down. This is a property of machine
images, not of this image.

## The fix: a plain disk image

A *disk* image carries only the disk contents:

```
architecture, diskSizeGb, guestOsFeatures, sourceType
```

No machine type. No accelerators. No disk interface — interface is an
*attachment* property, chosen when a disk is attached, so a disk image boots
any family. Verified against `debian-12`, whose describe output has none of
the three fields.

It can be captured from a **running** instance:

```bash
gcloud compute images create attnbench-env-v5 \
    --source-disk=<BOOT_DISK> \
    --source-disk-zone=<ZONE> \
    --force \
    --family=attnbench-env
```

`--force` is what allows capture without stopping the instance — by default
image creation refuses a disk attached to a running VM. Capturing from a
running instance also avoids the stop/start hazard that re-arms the launcher's
shutdown cap from a fresh boot time.

Then the whole seed/snapshot/attach sequence collapses to one create:

```bash
gcloud compute instances create attnbench-a100-... \
    --zone=asia-southeast1-c \
    --machine-type=a2-ultragpu-1g \
    --provisioning-model=SPOT \
    --image=attnbench-env-v5 --image-project=research-507316 \
    --boot-disk-size=200 --boot-disk-type=pd-balanced
```

No `--accelerator` — `a2-ultragpu-1g` includes its A100 in the machine type.
No `--boot-disk-interface` — with a disk image gcloud creates the boot disk, so
it picks an interface the target family accepts.

### Do it in the next G2 session

1. Boot a G2 session normally (same family, so the current image works).
2. Early, while the environment is known good, run the `images create` above
   against the instance's own boot disk with `--force`.
3. Verify: `gcloud compute images describe attnbench-env-v5` must show **no**
   `machineType` and **no** `guestAccelerators`.
4. Continue the session as planned. The capture is a background operation.

Images are **global**, so one capture serves every region — unlike a zonal
disk, which is the constraint that broke the seed plan.

Keep the machine image until a disk image has actually booted an A2. Two
artifacts for one environment is a small storage cost against re-doing a
three-hour compile.

## What the launcher does until then

`gcp_launch_compile_session.sh` refuses a cross-family launch before attempting
any create: it reads the image's `sourceInstanceProperties.machineType`,
compares families, and if they differ prints all three inherited properties and
points here. `GCP_ALLOW_CROSS_FAMILY_IMAGE=1` overrides it — needed because a
plain disk image records no machine type, so the check cannot tell that the fix
has been applied; absence of the field is treated as "do not block".

## The process note

Attempt 3 was chosen after checking `c3` against the NVME family list and
against `pd-balanced`, and rejecting `n4` and `c4` because both require
hyperdisk-balanced. Three constraints checked. The fourth — `guestAccelerators`
— was not, **despite it being the first inherited property that failed that
morning**.

The pattern is checking the constraints most recently learned rather than
enumerating the ones that apply. It is not a new failure mode; it is why the
structural answer is a preflight that enumerates all three every time, rather
than a resolution to be more careful.
