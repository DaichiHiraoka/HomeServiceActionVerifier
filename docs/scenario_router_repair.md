# Router Repair Scenario

## Overview

`router_repair` models a permitted home visit for checking a Wi-Fi router communication problem. The worker is allowed to enter, inspect the router shelf, handle router-related cables and adapters, photograph router labels or damage, use worker-owned tools, return worker-owned tools to the tool bag, and exit.

The system does not decide theft or crime. It marks event-level behavior as `normal`, `review`, `suspicious`, or `high_risk` for later human review.

## Work Order

The work order is stored at `configs/scenarios/router_repair.json`.

It defines:

- authorized zones: `entrance`, `work_area`, `router_shelf`
- forbidden zones: `private_desk`, `private_drawer`, `bedroom`
- target objects: `router`, `lan_cable`, `power_adapter`, `wall_socket`
- worker-owned objects: `tool_bag`, `screwdriver`, `tester`, `worker_phone`
- resident private objects: `wallet`, `key`, `document`, `medicine`, `resident_phone`
- allowed photo targets: `router_label`, `damaged_cable`, `repair_area`

## Normal Actions

- enter from the entrance
- move to the router shelf
- inspect the router
- unplug or plug LAN cables
- inspect the power adapter
- photograph the router label or damaged cable
- use worker-owned tools
- return worker-owned tools to the worker tool bag
- exit after work

## Suspicious Or High-Risk Actions

- approach a private desk
- open a private drawer
- pick up documents, keys, wallet, medicine, or resident phone
- photograph private documents
- place resident-owned objects into a worker container
- enter a forbidden zone

## Same Action Different Context Pairs

| pair_id | normal context | risky context |
| --- | --- | --- |
| `bag_context` | worker screwdriver into worker tool bag | resident key into worker tool bag |
| `photo_context` | photograph router label | photograph private document |
| `drawer_context` | task-related storage, when annotated | private desk drawer |

## Recording Notes

Keep camera framing fixed when using `configs/zones/router_repair_zones.json`. The zone coordinates are initial placeholders and should be calibrated to the real camera view before collecting final data.

Do not collect or infer personal identity, face identity, age, gender, body type, or clothing-derived attributes. The experiment only needs event timing, zone, action, object class, and ownership context.

## Annotation Format

Annotations are JSONL, one event per line. `label` is loaded as `ground_truth_label`.

Required fields are `event_id`, `start_sec`, `end_sec`, and `action`. Optional fields include `zone`, `object_class`, `object_owner`, `container_class`, `container_owner`, `target_object`, `same_action_pair_id`, and `notes`.

Example file:

```text
data/real/router_trial_001_annotations.example.jsonl
```
