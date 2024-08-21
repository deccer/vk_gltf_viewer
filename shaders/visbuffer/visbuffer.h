#ifndef SHADERS_VISBUFFER_H
#define SHADERS_VISBUFFER_H

#if !defined(__cplusplus)
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#endif

#include "common.h"
#include "mesh_common.h"
#include "resource_table.h"

SHADER_NAMESPACE_BEGIN

SHADER_CONSTANT uint32_t triangleBits = 7; // Enough to fit 127 unique triangles
SHADER_CONSTANT uint32_t drawIndexBits = 32 - triangleBits; // 25 bits for the drawIndex, which limits us
													  // to 33'554'432 meshlets for rendering.
SHADER_CONSTANT uint32_t visbufferClearValue = ~0U;

#if defined(SHADER_GLSL)
layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer materials_ref {
	material_t materials[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer meshlet_draws_ref {
	meshlet_draw_t draws[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer transforms_ref {
	fmat4 transforms[];
};

layout(buffer_reference, scalar, buffer_reference_align = 8) restrict readonly buffer primitives_ref {
	primitive_t primitives[];
};
#endif

struct VisbufferPushConstants {
	BUFFER_REF(meshlet_draws_ref, meshlet_draw_t) drawBuffer MEMBER_INIT(0);
	uint32_t meshletDrawCount MEMBER_INIT(0);

	BUFFER_REF(transforms_ref, fmat4) transformBuffer MEMBER_INIT(0);
	BUFFER_REF(primitives_ref, primitive_t) primitiveBuffer MEMBER_INIT(0);
	BUFFER_REF(CameraBuffer, Camera) cameraBuffer MEMBER_INIT(0);
	BUFFER_REF(materials_ref, material_t) materialBuffer MEMBER_INIT(0);

	ResourceTableHandle depthPyramid MEMBER_INIT(invalidHandle);
};

struct VisbufferResolvePushConstants {
	ResourceTableHandle visbufferHandle MEMBER_INIT(invalidHandle);
	ResourceTableHandle outputImageHandle MEMBER_INIT(invalidHandle);

	BUFFER_REF(meshlet_draws_ref, meshlet_draw_t) drawBuffer MEMBER_INIT(0);
	BUFFER_REF(primitives_ref, primitive_t) primitiveBuffer MEMBER_INIT(0);
	BUFFER_REF(materials_ref, material_t) materialBuffer MEMBER_INIT(0);
};

#if defined(SHADER_METAL)
struct visbuffer_data {
	uint32_t draw_index : drawIndexBits;
	uint32_t primitive_id : triangleBits;

	[[clang::always_inline]] constexpr visbuffer_data(uint32_t value) :
		draw_index(value >> triangleBits), primitive_id(value & ((1 << triangleBits) - 1)) {}
	[[clang::always_inline]] constexpr visbuffer_data(uint32_t draw_index, uint32_t primitive_id) :
		draw_index(draw_index), primitive_id(primitive_id) {}

	[[clang::always_inline]] constexpr bool is_valid() thread {
		return uint32_t(*this) != visbufferClearValue;
	}
	[[clang::always_inline]] constexpr bool is_valid() threadgroup_imageblock {
		return uint32_t(*this) != visbufferClearValue;
	}

	[[clang::always_inline]] constexpr explicit operator uint32_t() const thread {
		return (draw_index << triangleBits) | primitive_id;
	}
	[[clang::always_inline]] constexpr explicit operator uint32_t() const threadgroup_imageblock {
		return (draw_index << triangleBits) | primitive_id;
	}
};
static_assert(sizeof(visbuffer_data) == sizeof(uint32_t), "the visbuffer type needs to be exactly 32-bits.");
#elif defined(SHADER_GLSL)
uint32_t packVisBuffer(uint32_t drawIndex, uint32_t primitiveId) {
	return (drawIndex << triangleBits) | primitiveId;
}

void unpackVisBuffer(uint32_t visBuffer, out uint32_t drawIndex, out uint32_t primitiveId) {
	primitiveId = visBuffer & ((1 << triangleBits) - 1);
	drawIndex = visBuffer >> triangleBits;
}
#endif

SHADER_NAMESPACE_END
#endif
