#pragma once

#include <exception>
#include <string>

namespace pj {
class ProtoJitError : public std::exception {
 public:
  ProtoJitError() : error_("unknown") {}
  ProtoJitError(std::string&& error) : error_(error) {}

  const char* what() const noexcept override { return error_.c_str(); }

 private:
  const std::string error_;
};

class CompilerUserError : public ProtoJitError {
 public:
  CompilerUserError(std::string&& error) : ProtoJitError(std::move(error)) {}
};

class IssueError : public ProtoJitError {
 public:
  IssueError(intptr_t issue)
      : ProtoJitError(std::string("Not implemented. See: ") +
                      "https://ny5-github-v01.oldmissioncapital.com/"
                      "sjindel/protojit/issues/" +
                      std::to_string(issue)) {}
};

class InternalError : public ProtoJitError {
 public:
  InternalError(std::string&& error) : ProtoJitError(std::move(error)) {}
};

}  // namespace pj
