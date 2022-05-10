#include "variant_outlining.hpp"
#include "span.hpp"

namespace pj {
using namespace mlir;
using namespace types;

ValueType VariantOutlining::tryOutlineVariant(Type type, PathAttr path) {
  if (path.empty()) {
    return {};
  }

  if (auto var = type.dyn_cast<InlineVariantType>()) {
    // Modify the name so that the final identifier starts with a ^ to
    // disambiguate it from the InlineVariant. For example, A::B::C becomes
    // A::B::^C
    assert(var.name().size() > 0);
    std::vector<llvm::StringRef> name{var.name().begin(),
                                      std::prev(var.name().end())};
    std::string last = "^" + var.name().back().str();
    name.push_back(last);
    auto outline = OutlineVariantType::get(&ctx_, types::TypeDomain::kInternal,
                                           Name{&name[0], name.size()});

    // term_offset is left unset because it depends on the final generated
    // head sizes of all types coming after it. It will be set in the end in
    // OutlineVariantOffsetGeneration. tag_alignment and term_alignment will be
    // similarly set in WireLayout.
    outline.setTypeData({.terms = var->terms,
                         .tag_width = var->tag_width,
                         .tag_alignment = Width::None(),
                         .term_offset = Bytes(0),
                         .term_alignment = Width::None()});
    return outline;
  }

  if (auto str = type.dyn_cast<StructType>()) {
    for (uintptr_t i = 0; i < str->fields.size(); ++i) {
      const StructField& f = str->fields[i];
      if (auto outline = tryOutlineVariant(f.type, path.into(f.name))) {
        replaceStructField(str, i, outline);
        return str;
      }
    }
  }

  return {};
}

bool VariantOutlining::run(Protocol& proto) {
  if (auto head = tryOutlineVariant(proto.head, path_)) {
    proto.head = head;
    return true;
  }
  return false;
}

void OutlineVariantOffsetGeneration::incrementTermOffset(Width val) {
  assert(outline_);
  auto data = OutlineVariant(outline_);
  data.term_offset += val;
  outline_.setTypeData(data);
}

void OutlineVariantOffsetGeneration::run(Type type) {
  if (auto var = type.dyn_cast<OutlineVariantType>()) {
    // Must run after WireLayout
    assert(var.type_domain() == TypeDomain::kWire);
    outline_ = var;
    incrementTermOffset(var.headSize());
    return;
  }

  if (auto str = type.dyn_cast<StructType>()) {
    if (!str->outline_variant) {
      return;
    }

    bool outline_reached = false;
    Width offset = str.headSize();
    for (const StructField& f : str->fields) {
      // Record the offset of the field following the one containing the outline
      // variant...
      if (!outline_reached && outline_) {
        outline_reached = true;
        offset = f.offset;
      }

      run(f.type);
    }

    // ... so that the distance from there to the end of the struct can be
    // added to the outline term offset.
    incrementTermOffset(str.headSize() - offset);
    return;
  }
}

bool OutlineVariantOffsetGeneration::run(Protocol& proto) {
  run(proto.head);

  // Update the outline variant's term offset and the protocol's buffer offset
  // to be properly aligned.
  if (outline_) {
    auto proto_align = proto.head.headAlignment().bytes();
    auto term_align = outline_->term_alignment.bytes();
    assert(proto_align >= term_align && proto_align % term_align == 0);

    auto old_size = proto.head.headSize();
    auto new_size = RoundUp(old_size, outline_->term_alignment);
    auto diff = new_size - old_size;

    incrementTermOffset(diff);
    proto.buffer_offset = diff;
  }

  return static_cast<bool>(outline_);
}

}  // namespace pj
