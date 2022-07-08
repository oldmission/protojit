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
