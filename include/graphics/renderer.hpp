#pragma once

#include <memory>
#include <span>

#include <GLFW/glfw3.h>

#include <mesh_common.h>
#include <resource_table.h>

#include <fastgltf/types.hpp>

namespace graphics {
	class renderer;

	using index_t = std::uint32_t; // TODO: Support dynamic index bit width?

	/** The amount of frames we render ahead, regardless of what the windowing system or GPU supports */
	static constexpr std::uint32_t frame_overlap = 3;

	class buffer_t {};

	class sampler_t {};
	class image_t {};
	class texture_t {
		shaders::resource_table_handle_t handle = shaders::invalid_handle;

	public:
		explicit texture_t(const shaders::resource_table_handle_t handle) noexcept : handle(handle) {}
		virtual ~texture_t() noexcept = default;

		[[nodiscard]] shaders::resource_table_handle_t get_handle() const noexcept {
			return handle;
		}
	};

	class mesh_t {};

	using material_index = std::uint32_t;
	using instance_index = std::uint32_t;

	class scene_t {
	public:
		scene_t() noexcept = default;
		virtual ~scene_t() noexcept = default;

		[[nodiscard]] virtual instance_index add_mesh_instance(std::shared_ptr<mesh_t> mesh) = 0;
		virtual void update_transform(instance_index instance, glm::fmat4x4 transform) = 0;
	};

	/**
	 * The abstracted renderer interface.
	 */
	class renderer : public std::enable_shared_from_this<renderer> {
	public:
		renderer() noexcept = default;
		virtual ~renderer() noexcept = default;

		[[nodiscard]] static std::shared_ptr<renderer> create_renderer(GLFWwindow* window);

		[[nodiscard]] virtual std::unique_ptr<buffer_t> create_unique_buffer() = 0;
		[[nodiscard]] virtual std::shared_ptr<buffer_t> create_shared_buffer() = 0;

		[[nodiscard]] virtual material_index get_default_material_index() const noexcept {
			return 0;
		}
		[[nodiscard]] virtual material_index create_material(shaders::material_t material) = 0;

		[[nodiscard]] virtual std::shared_ptr<mesh_t> create_shared_mesh(
				std::span<const glm::fvec3> positions, std::span<const shaders::vertex_t> vertices,
				std::array<std::span<const glm::fvec2>, shaders::max_uv_sets> uvs, std::span<const index_t> indices,
				glm::fvec3 aabb_center, glm::fvec3 aabb_extents,
				material_index material_index) = 0;

		[[nodiscard]] virtual std::shared_ptr<scene_t> create_shared_scene() = 0;

		[[nodiscard]] virtual std::shared_ptr<image_t> create_shared_image(
			std::span<std::byte> imageData, glm::u32vec2 extents) = 0;

		[[nodiscard]] virtual std::shared_ptr<sampler_t> get_default_sampler() = 0;
		[[nodiscard]] virtual std::shared_ptr<sampler_t> create_shared_sampler(
			const fastgltf::Sampler& sampler) = 0; // TODO

		[[nodiscard]] virtual std::shared_ptr<texture_t> create_shared_texture(
			std::shared_ptr<image_t> image, std::shared_ptr<sampler_t> sampler) = 0;

		/**
		 * If this returns false, the window might be minimised or being resized, forcing us to pause rendering shortly.
		 * In that case, glfwWaitEvents should be used.
		 */
		[[nodiscard]] virtual bool can_render() = 0;

		virtual void update_resolution(glm::u32vec2 resolution) = 0;
		/**
		 * Returns the resolution at which this renders the scene. Note that this might not be
		 * the same resolution the window has, since the renderer might use some upscaling
		 * technique.
		 */
		[[nodiscard]] virtual glm::u32vec2 get_render_resolution() const noexcept = 0;

		virtual void prepare_frame(std::size_t frame_index) = 0;
		virtual bool draw(std::size_t frame_index, scene_t& scene,
						  const shaders::camera_t& camera, float dt) = 0;
	};
}
