#ifndef SHADERS_SRGB_H
#define SHADERS_SRGB_H

#include "common.h"
#include "relational.h"
SHADER_NAMESPACE_BEGIN

/** sRGB conversion functions shamelessly stolen from https://gamedev.stackexchange.com/a/194038/159451 */

FUNCTION_INLINE float from_linear(float linearRGB) {
	bool cutoff = linearRGB < 0.0031308f;
	float higher = 1.055f * pow(linearRGB, 1.f / 2.4f) - 0.055f;
	float lower = linearRGB * 12.92f;

	return mix(higher, lower, cutoff);
}

FUNCTION_INLINE float to_linear(float sRGB) {
	bool cutoff = sRGB < 0.04045f;
	float higher = pow((sRGB + 0.055f) / 1.055f, 2.4f);
	float lower = sRGB / 12.92f;

	return mix(higher, lower, cutoff);
}

#if defined(SHADER_METAL)
// Converts a color from linear light gamma to sRGB gamma
FUNCTION_INLINE f16vec4 from_linear(f16vec4 linearRGB) {
	bvec3 cutoff = lessThan(linearRGB.rgb, f16vec3(0.0031308f));
	f16vec3 higher = f16vec3(1.055f) * pow(linearRGB.rgb, f16vec3(1.f / 2.4f)) - f16vec3(0.055f);
	f16vec3 lower = linearRGB.rgb * f16vec3(12.92f);

	return f16vec4(mix(higher, lower, f16vec3(cutoff)), linearRGB.a);
}

// Converts a color from sRGB gamma to linear light gamma
FUNCTION_INLINE f16vec4 to_linear(f16vec4 sRGB) {
	bvec3 cutoff = lessThan(sRGB.rgb, f16vec3(0.04045f));
	f16vec3 higher = pow((sRGB.rgb + f16vec3(0.055f)) / f16vec3(1.055f), f16vec3(2.4f));
	f16vec3 lower = sRGB.rgb / f16vec3(12.92f);

	return f16vec4(mix(higher, lower, f16vec3(cutoff)), sRGB.a);
}
#endif

// Converts a color from linear light gamma to sRGB gamma
FUNCTION_INLINE fvec4 from_linear(fvec4 linearRGB) {
	bvec3 cutoff = lessThan(linearRGB.rgb, fvec3(0.0031308f));
	fvec3 higher = fvec3(1.055f) * pow(linearRGB.rgb, fvec3(1.f / 2.4f)) - fvec3(0.055f);
	fvec3 lower = linearRGB.rgb * fvec3(12.92f);

	return fvec4(mix(higher, lower, fvec3(cutoff)), linearRGB.a);
}

// Converts a color from sRGB gamma to linear light gamma
FUNCTION_INLINE fvec4 to_linear(fvec4 sRGB) {
	bvec3 cutoff = lessThan(sRGB.rgb, fvec3(0.04045f));
	fvec3 higher = pow((sRGB.rgb + fvec3(0.055f)) / fvec3(1.055f), fvec3(2.4f));
	fvec3 lower = sRGB.rgb / fvec3(12.92f);

	return fvec4(mix(higher, lower, fvec3(cutoff)), sRGB.a);
}

SHADER_NAMESPACE_END
#endif
