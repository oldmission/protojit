# Variant optimization testing
This branch is for manually testing that variant term optimizations are
backwards compatible. Build //demo:demo on HEAD (v0.2) and on HEAD~1
(v0.1) to get a version that has the int shortening optimization, and
one that doesn't. Only the "v1" protocol has an int eligible for the
optimization, so --proto-version=v1 should be used for this testing.

Check that both v0.1 and v0.2 can read the output of the following, and
change the value of `short_int` in demo/main.cpp from 42 to a number
larger than 255 to ensure that the correct terms are really being used:
* v0.1 writes using its own plan (--mode=write)
* v0.2 writes using the v0.1 schema (--mode=writeexisting)
* v0.2 writes using its own plan
* v0.1 writes using the v0.2 schema

This makes for a total of 16 separate cases to test (4 writing options,
2 reading options, and 2 `short_int` values).

# Self-describing format
The canonical self-describing record format has the following layout:

```
┌────────────────────────────────────┬───────────────────────┬─────────┬─────┐
│               Schema               │                       │         │     │
├──────────────────────┬─────────────┤ 8 byte message length │ Message │ ... │
│ 8 byte schema length │ Schema data │                       │         │     │
└──────────────────────┴─────────────┴───────────────────────┴─────────┴─────┘
```

The entire schema, including the 8 byte schema length, is included in
`getProtoSize`, written out by `encodeProto`, and required by `decodeProto`.

# Testing
Unit tests can be found in the `test` directory and run using `bazel test`.

See branch variant-opt-test for manual testing of variant term optimizations'
backwards and forwards compatibility. In the long-term, we will have a more
dedicated strategy for testing compatibility.
