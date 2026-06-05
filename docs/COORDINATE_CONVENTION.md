# Coordinate Convention

This document defines the coordinate contract for all future Fast-dVLM /
GUI-Owl SFT, KD, evaluation, and AndroidWorld execution.

## Locked Contract

```text
name: guiowl_norm1000_xy
order: [x, y]
range: 0..1000
origin: top-left
executor maps: normalized 0..1000 -> device pixels
```

Every `mobile_use` target must use this convention.

Click:

```json
{
  "name": "mobile_use",
  "arguments": {
    "action": "click",
    "coordinate": [521, 843]
  }
}
```

Swipe:

```json
{
  "name": "mobile_use",
  "arguments": {
    "action": "swipe",
    "coordinate": [658, 888],
    "coordinate2": [745, 361]
  }
}
```

## Conversion From Pixels

If a source dataset stores pixels, convert before writing `target_json`:

```text
x_norm1000 = round(x_pixel / screen_width  * 1000)
y_norm1000 = round(y_pixel / screen_height * 1000)
```

For swipe, convert both endpoints.

After conversion:

```text
0 <= x <= 1000
0 <= y <= 1000
```

Rows outside range must be either fixed with a documented reason or dropped.

## Execution Conversion

Only the executor should convert normalized coordinates to pixels:

```text
x_pixel = round(x_norm1000 / 1000 * screen_width)
y_pixel = round(y_norm1000 / 1000 * screen_height)
```

Training targets should not contain screen-specific pixel coordinates.

## Required Dataset Validation

Every curated dataset must produce a validation report with:

- number of rows
- number of parsed `mobile_use` rows
- action-type distribution
- coordinate mode per source
- before/after coordinate examples
- out-of-range coordinate count
- suspected `[y, x]` rows
- click/swipe/type/system_button/terminate counts
- duplicates and capped repeated goals

Minimum acceptance:

```text
parse failures: documented and bounded
out-of-range coords: 0 after final conversion
coordinate mode: guiowl_norm1000_xy
```

## Historical Failure

A previous Gmail SFT path mixed coordinate systems:

```text
gmail rows: pixel coordinates
general rows: normalized 0..1000
AndroidWorld executor: interpreted model output as normalized 0..1000
```

This created compressed coordinates during AndroidWorld testing. Example:

```text
pixel x in a 412-wide screenshot -> model emits around 0..412
executor interprets 412 as 41.2% of screen width, not the original pixel
```

This is a major grounding failure mode. It is not the only issue, but it is a
required audit point before every future training run.

## GUI-Owl Compatibility

The teacher target for current distillation is GUI-Owl-1.5 behavior. Therefore:

- match GUI-Owl normalized coordinates
- match GUI-Owl `mobile_use` schema
- use `[x, y]`, not `[y, x]`
- keep strict/repaired metrics separate

If a future teacher emits a different schema, convert it to this canonical
schema before SFT unless the tokenizer/policy interface is deliberately changed.
