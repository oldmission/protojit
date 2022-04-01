#ifndef PJ_DEFER_HPP
#define PJ_DEFER_HPP

#include <functional>

namespace pj {

// Hold a nullary functor and call it when destroyed. Take care to avoid
// allocation in the functor construction if necessary via
// std::reference_wrapper or DEFER.
struct Defer {
  // Hold a null functor, do nothing on destruction. This can be used to
  // disable a Defer, e.g. `Defer x(...); ...; x = {}`.
  Defer() {}

  ~Defer() {
    if (deferred) {
      deferred();
    }
  }

  explicit Defer(std::function<void()>&& f) : deferred(std::move(f)) {}

  std::function<void()> deferred;
};

// Use __COUNTER__ to manufacture a fresh variable name for the closure.
// We must put it on the stack to specify its storage and use
// std::reference_wrapper to prevent it from being copied into the heap
// by the std::function constructor.
#define DEFER_COUNTER(CODE, CTR, STRUCT_NAME)                   \
  auto deferred_capture_##CTR = [&]() { CODE; };                \
  pj::Defer STRUCT_NAME{                                        \
      std::reference_wrapper<decltype(deferred_capture_##CTR)>( \
          deferred_capture_##CTR)};

// We need to pass the arguments to DEFER_COUNTER (specifically CTR) through
// an extra "expander" macro to ensure that the counter is evaluated *before*
// being substituted into DEFER_COUNTER. Otherwise, it would be evaluated twice,
// and the statement defining closure variable would see a different value of
// the counter than the statement which references that variable.
#define DEFER_UNNAMED_INNER(CODE, CTR) DEFER_COUNTER(CODE, CTR, deferred_##CTR)
#define DEFER_UNNAMED_EXPANDER(CODE, CTR) DEFER_UNNAMED_INNER(CODE, CTR)
#define DEFER_NAMED_EXPANDER(CODE, CTR, STRUCT_NAME) \
  DEFER_COUNTER(CODE, CTR, STRUCT_NAME)

// Helper macros which are the targets of the variadic macro dispatch below.
// DEFER_UNNAMED is called for anonymous Defer declarations, whereas DEFER_NAMED
// is called when the Defer variable name is specified.
#define DEFER_UNNAMED(X) DEFER_UNNAMED_EXPANDER(X, __COUNTER__)
#define DEFER_NAMED(N, X) DEFER_NAMED_EXPANDER(X, __COUNTER__, N)
#define DEFER_ARGS_DISPATCHER(_1, _2, NAME, ...) NAME

// DEFER(<code>) will execute <code> when the enclosing block ends (in an
// RAII-like manner, e.g. before destructors for variables declared earlier but
// after destructors for variables declared later).
//
// For example:
// {
//   int x = 0;
//   DEFER(std::cout << x << "\n")
//   x = 3;
// }
//
// will print "3". For example, DEFER can be used for ad-hoc resource cleanup,
// e.g. as `DEFER(file->Close())` or `DEFER(ptr.reset())`.
//
// A second form of DEFER exists to allow disabling the deferred action in
// certain code paths. For example:
//
// {
//   int x = 0;
//   DEFER(print_x, std::cout << x << "\n");
//   x = 3;
//   print_x = {};
// }
//
// will not print anything. DEFER(<name>, <code>) defines a Defer variable named
// <name>, and assigning an empty Defer ({}) to this variable will remove the
// action originally attached to that variable.
//
// Any references in <code> to variables defined in the enclosing scope are
// taken by reference, and no copying or allocation is performed for the
// captured variables or <code>'s context itself.
#define DEFER(...) \
  DEFER_ARGS_DISPATCHER(__VA_ARGS__, DEFER_NAMED, DEFER_UNNAMED)(__VA_ARGS__)

}  // namespace pj

#endif  // PJ_DEFER_HPP
