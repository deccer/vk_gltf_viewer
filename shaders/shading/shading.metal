#include <metal_stdlib>

#include "shading.h"
#include "mesh_common.h"
#include "srgb.h"

namespace mtl = metal;

float3 reconstruct_position_from_depth(constant shaders::fmat4& inv, float2 uv, float depth) {
	float4 clip_pos(uv * 2.f - 1.f, depth, 1.f);
	clip_pos.y = -clip_pos.y;
	auto world_pos = inv * clip_pos;
	return world_pos.xyz / world_pos.w;
}

/// Simple Lambertian diffuse BRDF
half3 diffuse_brdf(half3 color) {
	// (1 / pi) * color
	return M_1_PI_H * color;
}

/// The Heaviside function: 1 if x > 0 and 0 if x ⇐ 0
template <typename T>
T heaviside_function(T value) {
	return mtl::select(T(0), T(1), value > T(0));
}

/// The Trowbridge-Reitz/GGX microfacet distribution optimised for FP16.
/// See https://google.github.io/filament/Filament.html#materialsystem/specularbrdf/normaldistributionfunction(speculard)
half microfacet_distribution(half alpha, half NdotH, half3 NxH) {
	auto a = NdotH * alpha;
	auto k = alpha / (mtl::dot(NxH, NxH) + a * a);
	auto d = k * k * M_1_PI_H;
	return mtl::min(d, MAXHALF);
}

half visibility_function(half alpha, half NdotL, half NdotV) {
	auto alpha_squared = mtl::pow(alpha, 2.h);

	auto ggxv = NdotL * mtl::sqrt(mtl::pow(NdotV, 2.h) * (1.h - alpha_squared) + alpha_squared);
	auto ggxl = NdotV * mtl::sqrt(mtl::pow(NdotL, 2.h) * (1.h - alpha_squared) + alpha_squared);

	auto ggx = ggxv + ggxl;
	if (ggx > 0.00001h) // Avoids div by 0 and returning infinity
		return 0.5h / ggx;
	return 0.h;
}

/// A Cook-Torrance specular term with a GGX normal distribution and a Smith-GGX height-correlated visibility function.
half3 specular_brdf(half alpha, half NdotL, half NdotV, half NdotH, half3 NxH) {
	return microfacet_distribution(alpha, NdotH, NxH) * visibility_function(alpha, NdotL, NdotV);
}

/// Fresnel term for the dielectric BRDF where base is the diffuse component, and
/// layer the specular, which get combined based on the index of refraction.
half3 fresnel_mix(half ior, half3 base, half3 layer, half VdotH) {
	auto f0 = mtl::pow((1-ior)/(1+ior), 2.h);
	auto fr = f0 + (1 - f0) * mtl::pow(1 - mtl::abs(VdotH), 5.h);
	return mtl::mix(base, layer, fr);
}

half3 conductor_fresnel(half3 f0, half3 bsdf, half VdotH) {
	return bsdf * (f0 + (1 - f0) * mtl::pow(1 - mtl::abs(VdotH), 5.h));
}

struct gbuffer_tile_data {
	packed_half4 color [[raster_order_group(0)]];
	mtl::rgba8unorm<half4> albedo [[raster_order_group(0)]];
	mtl::rg16unorm<float2> normal [[raster_order_group(0)]];
	packed_half2 metallic_roughness [[raster_order_group(0)]];
};

[[kernel]] void gbuffer_shading(
		mtl::imageblock<gbuffer_tile_data> tile_data [[alias_implicit_imageblock]],
		constant const shaders::camera_t& camera [[buffer(0)]],
		mtl::depth2d<float> depth_texture [[texture(0)]],
		ushort2 local_tid [[thread_position_in_threadgroup]],
		ushort2 tid [[thread_position_in_grid]]) {
	threadgroup_imageblock gbuffer_tile_data* data = tile_data.data(local_tid);

	auto depth = depth_texture.read(tid);
	if (depth == 0.f)
		return;

	auto uv = (float2(tid) + 0.5f) / float2(depth_texture.get_width(), depth_texture.get_height());
	auto position = reconstruct_position_from_depth(camera.invViewProjection, uv, depth);

	auto light = mtl::normalize(half3(1.h, 1.h, 1.h));
	auto view = half3(mtl::normalize(camera.position - position));
	auto halfv = mtl::normalize(light + view);

	auto albedo = half4(data->albedo).xyz;
	auto roughness = mtl::pow(data->metallic_roughness.x, 2.h);
	auto normal = half3(shaders::normal_decode(data->normal));

	auto NdotV = mtl::abs(mtl::dot(normal, view)) + 1E-5h;
	auto NdotH = mtl::saturate(mtl::dot(normal, halfv));
	auto NdotL = mtl::saturate(mtl::dot(normal, light));
	auto LdotH = mtl::saturate(mtl::dot(light, halfv));
	auto VdotH = mtl::dot(view, halfv);

	constexpr auto ambient = half3(0.1h);
	constexpr auto light_intensity = 2.5h;

	auto diffuse = light_intensity * NdotL * diffuse_brdf(albedo);
	auto specular = light_intensity * NdotL * specular_brdf(roughness, NdotL, NdotV, NdotH, mtl::cross(normal, halfv));
	auto dielectric_brdf = fresnel_mix(1.5h, diffuse, specular, VdotH);
	auto metal_brdf = conductor_fresnel(albedo, specular, VdotH);
	auto final = mtl::mix(dielectric_brdf, metal_brdf, data->metallic_roughness.y);

	data->color = shaders::from_linear(half4(ambient * albedo + final, 1.h));
}
