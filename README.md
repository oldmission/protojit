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
