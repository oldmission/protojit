/**
 * Defines the C Runtime API used by the code generated by pjc
 */

#ifndef PJ_RUNTIME_H
#define PJ_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef intptr_t Bits;

enum PJSign { PJ_SIGN_SIGNED, PJ_SIGN_UNSIGNED, PJ_SIGN_SIGNLESS };
enum PJTypeDomain { PJ_TYPE_DOMAIN_HOST, PJ_TYPE_DOMAIN_WIRE };
enum PJReferenceMode { PJ_REFERENCE_MODE_POINTER, PJ_REFERENCE_MODE_OFFSET };

typedef struct PJContext PJContext;
typedef struct PJUnitType PJUnitType;
typedef struct PJIntType PJIntType;
typedef struct PJStructField PJStructField;
typedef struct PJStructType PJStructType;
typedef struct PJAnyType PJAnyType;
typedef struct PJTerm PJTerm;
typedef struct PJInlineVariantType PJInlineVariantType;
typedef struct PJArrayType PJArrayType;
typedef struct PJVectorType PJVectorType;
typedef struct PJProtocol PJProtocol;

typedef struct PJHandler {
  const char* name;
  const void* function;
} PJHandler;

const PJUnitType* PJCreateUnitType(PJContext* c);

const PJIntType* PJCreateIntType(PJContext* c, Bits width, Bits alignment,
                                 PJSign sign);

const PJStructField* PJCreateStructField(const char* name, const void* type,
                                         Bits offset);

const PJStructType* PJCreateStructType(PJContext* c, uintptr_t name_size,
                                       const char* name[],
                                       PJTypeDomain type_domain,
                                       uintptr_t num_fields,
                                       const PJStructField* fields[], Bits size,
                                       Bits alignment);

const PJAnyType* PJCreateAnyType(PJContext* c, Bits data_ref_offset,
                                 Bits data_ref_width, Bits type_ref_offset,
                                 Bits type_ref_width, Bits size,
                                 Bits alignment);

const PJTerm* PJCreateTerm(const char* name, const void* type, uint64_t tag);

const PJInlineVariantType* PJCreateInlineVariantType(
    PJContext* c, uintptr_t name_size, const char* name[],
    PJTypeDomain type_domain, uintptr_t num_terms, const PJTerm* terms[],
    Bits term_offset, Bits term_size, Bits tag_offset, Bits tag_size, Bits size,
    Bits alignment);

const PJArrayType* PJCreateArrayType(PJContext* c, const void* type,
                                     uint64_t length, Bits elem_size,
                                     Bits alignment);

const PJVectorType* PJCreateVectorType(
    PJContext* c, const void* type, intptr_t min_length, intptr_t max_length,
    intptr_t wire_min_length, intptr_t ppl_count, Bits length_offset,
    Bits length_size, Bits ref_offset, Bits ref_size,
    PJReferenceMode reference_mode, Bits inline_payload_offset,
    Bits inline_payload_size, Bits partial_payload_offset,
    Bits partial_payload_size, Bits size, Bits alignment,
    Bits outlined_payload_alignment);

const PJProtocol* PJPlanProtocol(PJContext* ctx, const void* head,
                                 const char* tag_path);

void PJAddEncodeFunction(PJContext* ctx, const char* name, const void* src,
                         const PJProtocol* protocol, const char* src_path);

void PJAddDecodeFunction(PJContext* ctx, const char* name,
                         const PJProtocol* protocol, const void* dest,
                         uintptr_t num_handlers, const PJHandler* handlers[]);

void PJAddSizeFunction(PJContext* ctx, const char* name, const void* src,
                       const PJProtocol* protocol, const char* src_path,
                       bool round_up);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // PJ_RUNTIME_H
