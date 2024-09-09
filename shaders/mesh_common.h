#ifndef SHADERS_MESH_COMMON_H
#define SHADERS_MESH_COMMON_H

#include "common.h"
#include "relational.h"

#if defined(SHADER_GLSL)
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

#if defined(SHADER_CPP)
#include <glm/gtx/vec_swizzle.hpp>
#endif

#include "resource_table.h"
SHADER_NAMESPACE_BEGIN

SHADER_CONSTANT uint32_t shadowMapCount = 4;

struct camera_t {
	ALIGN_AS(16) fmat4 prevViewProjection;
	ALIGN_AS(16) fmat4 prevOcclusionViewProjection;

	ALIGN_AS(16) fmat4 viewProjection;
	ALIGN_AS(16) fmat4 invViewProjection;
	ALIGN_AS(16) fmat4 occlusionViewProjection;
	SHADER_ARRAY(packed_fvec4, frustum, 6);

	packed_fvec3 position;
};

#if defined(SHADER_GLSL)
layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer camera_ref {
	camera_t camera;
};
#endif

// TODO: These are the optimal values for NVIDIA. What about the others?
#if defined(SHADER_GLSL)
SHADER_CONSTANT uint32_t maxVertices = 64;
SHADER_CONSTANT uint32_t maxPrimitives = 126;
SHADER_CONSTANT uint32_t maxMeshlets = 102;
#else
SHADER_CONSTANT uint32_t maxVertices = 64;
SHADER_CONSTANT uint32_t maxPrimitives = 128;
SHADER_CONSTANT uint32_t maxMeshlets = 32; // This should best be a multiple of the threadgroup size.
#endif

// This is essentially a replacement for gl_WorkGroupID.x, but one which can store any index
// between 0..256 instead of the linear requirement of the work group ID.
// For NVIDIA, we try to keep this structure below 108 bytes to keep it in shared memory.
// Therefore, this is exactly 106 bytes big (4 + 102 * 1).
struct TaskPayload {
	uint32_t baseID;
	uint8_t deltaIDs[maxMeshlets];
};

struct Meshlet {
	uint32_t vertexOffset;
	uint32_t triangleOffset;

	uint8_t vertexCount;
	uint8_t triangleCount;

	packed_fvec3 aabb_extents;
	packed_fvec3 aabb_center;
};

struct vertex_t {
#if VERTEX_BUFFER_NORMAL_ENCODING == 1
#if defined(SHADER_METAL)
	mtl::rg16snorm<float2> normal;
#else
	uint32_t normal;
#endif
#else
	packed_fvec3 normal;
#endif

#if VERTEX_BUFFER_NORMAL_ENCODING == 1
	uint32_t tangent;
#else
	packed_fvec4 tangent;
#endif

#if defined(SHADER_METAL)
	mtl::rgba8unorm<float4> color;
#else
	packed_u8vec4 color;
#endif
};

#if defined(SHADER_GLSL)
vec4 unpackVertexColor(in u8vec4 color) {
	return vec4(color) / 255.f;
}

vec3 unpackVertexNormal(in u8vec3 normal) {
	return normalize(vec3(normal) / 127.f - 1.f);
}
#endif

#if defined(SHADER_GLSL)
layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer vertex_indices_ref {
	uint vertexIndices[];
};
layout(buffer_reference, scalar, buffer_reference_align = 1) restrict readonly buffer primitive_indices_ref {
	uint8_t primitiveIndices[];
};
layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer positions_ref {
	vec3 positions[];
};
layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer meshlets_ref {
	Meshlet meshlets[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer vertices_ref {
	vertex_t vertices[];
};
layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer uvs_ref {
	vec2 uvs[];
};
#endif

struct meshlet_draw_t {
	uint32_t primitive_index;
	uint32_t meshlet_index;
	uint32_t transform_index;
};

// The maximum amount of UV sets is dictated by the maximum amount of textures a material
// can have. Whenever we add support for additional extensions that add more textures this needs
// to be increased accordingly.
SHADER_CONSTANT uint32_t max_uv_sets = 5;

struct primitive_t {
	BUFFER_REF(vertex_indices_ref, uint32_t) vertex_index_buffer MEMBER_INIT(0);
	BUFFER_REF(primitive_indices_ref, uint8_t) primitive_index_buffer MEMBER_INIT(0);
	BUFFER_REF(positions_ref, packed_fvec3) position_buffer MEMBER_INIT(0);
	BUFFER_REF(meshlets_ref, Meshlet) meshlet_buffer MEMBER_INIT(0);

	BUFFER_REF(vertices_ref, vertex_t) vertex_buffer MEMBER_INIT(0);
	SHADER_ARRAY(BUFFER_REF(uvs_ref, fvec2), uv_buffers, max_uv_sets);

	packed_fvec3 aabb_extents;
	packed_fvec3 aabb_center;

	uint32_t meshlet_count;
	uint32_t material_index;
};

struct primitive_transform_t {
	ALIGN_AS(16) fmat4 matrix;
	ALIGN_AS(16) fmat4 inverse_matrix;
};

struct material_texture_t {
	ResourceTableHandle index MEMBER_INIT(shaders::invalid_handle);
	uint32_t uv_set MEMBER_INIT(0);
	packed_fvec2 uv_offset MEMBER_INIT(fvec2(0.f));
	packed_fvec2 uv_scale MEMBER_INIT(fvec2(1.f));
	float uv_rotation MEMBER_INIT(0.f);
};

#if !defined(SHADER_GLSL)
enum alpha_mode_e : uint8_t {
	opaque,
	mask,
	blend
};
#else
#define alpha_mode_e uint8_t
#endif

struct material_t {
	packed_fvec4 albedo_factor;
	material_texture_t albedo;

	float metallic_factor;
	float roughness_factor;
	material_texture_t metallic_roughness;

	float normal_scale;
	material_texture_t normal;

	alpha_mode_e alpha_mode;
	float alpha_cutoff;
	SHADER_BOOL double_sided;
	float ior;
};

#if defined(SHADER_METAL) || defined(SHADER_GLSL)
FUNCTION_INLINE fvec2 transform_uv(material_texture_t texture, fvec2 uv) {
	fmat2 rotation_mat = fmat2(
		cos(texture.uv_rotation), -sin(texture.uv_rotation),
		sin(texture.uv_rotation), cos(texture.uv_rotation)
	);
	return rotation_mat * uv * texture.uv_scale + texture.uv_offset;
}
#endif

// Octahedron-normal vectors encoding for use in the GBuffer and for compressing normals & tangents
FUNCTION_INLINE fvec2 sign_not_zero(fvec2 v) {
	return fvec2(
		(v.x >= 0.f) ? +1.f : -1.f,
		(v.y >= 0.f) ? +1.f : -1.f);
}

#if !defined(SHADER_CPP)
FUNCTION_INLINE fvec2 normal_encode(fvec3 v) {
	const fvec2 p = v.xy * (1.f / (abs(v.x) + abs(v.y) + abs(v.z)));
	return (v.z <= 0.f)
		? ((1.f - abs(p.yx)) * sign_not_zero(p))
		: p;
}

FUNCTION_INLINE fvec3 normal_decode(fvec2 e) {
	fvec3 v = fvec3(e.xy, 1.f - abs(e.x) - abs(e.y));
	if (v.z < 0.f) {
		v.xy = (1.f - abs(v.yx)) * sign_not_zero(v.xy);
	}
	return normalize(v);
}

FUNCTION_INLINE fvec4 tangent_decode(uint32_t v) {
	const float sign = bool(v & 1) ? -1.f : +1.f;
#if defined(SHADER_METAL)
	const auto e = unpack_snorm2x16_to_float(v & ~1);
	return fvec4(normal_decode(e), sign);
#endif
	return fvec4(1.f); // TODO
}
#else
// TODO: Is there some way to share these definitions by introducing some swizzling magic for shared headers?
inline auto normal_encode(fvec3 v) {
	const auto p = xy(v) * (1.f / (abs(v.x) + abs(v.y) + abs(v.z)));
	return (v.z <= 0.f)
		? ((1.f - glm::abs(yx(p))) * sign_not_zero(p))
		: p;
}

inline auto normal_decode(fvec2 e) {
	auto v = fvec3(xy(e), 1.f - abs(e.x) - abs(e.y));
	if (v.z < 0.f) {
		xy(v) = (1.f - abs(yx(v))) * sign_not_zero(xy(v));
	}
	return normalize(v);
}

// the tangent encoding is effectively identical to the normal encoding, but
// this additionally stores the sign bit in the lowest bit of the encoded integer.
// this makes us loose a tiny bit of precision which should be effectively unnoticeable.
inline auto tangent_encode(fvec4 v) {
	const auto e = glm::packSnorm2x16(normal_encode(xyz(v)));
	const auto sign_bit = v.w >= 0.f ? 0 : 1;
	return (e & ~1) | (sign_bit);
}
#endif

SHADER_NAMESPACE_END
#endif
