#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

template <typename T, std::size_t N, typename Transform>
constexpr auto transform_array(const std::array<T, N>& input, Transform&& transform) {
	using U = std::remove_cvref_t<std::invoke_result_t<Transform, T&>>;
	std::array<U, N> output;
	std::transform(input.begin(), input.end(), output.begin(), transform);
	return output;
}
