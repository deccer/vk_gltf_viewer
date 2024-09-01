#pragma once

#include <memory>
#include <span>

#include <GLFW/glfw3.h>

#include <mesh_common.h>
#include <resource_table.h>

#include <fastgltf/types.hpp>

namespace graphics {
	class Renderer;

	using index_t = std::uint32_t; // TODO: Support dynamic index bit width?

	/** The amount of frames we render ahead, regardless of what the windowing system or GPU supports */
	static constexpr std::uint32_t frameOverlap = 3;

	class Buffer {};

	class Sampler {};
	class Image {};
	class Texture {
		shaders::ResourceTableHandle handle = shaders::invalid_handle;

	public:
		explicit Texture(const shaders::ResourceTableHandle handle) noexcept : handle(handle) {}
		virtual ~Texture() noexcept = default;

		[[nodiscard]] shaders::ResourceTableHandle getHandle() const noexcept {
			return handle;
		}
	};

	class Mesh {};

	using MaterialIndex = std::uint32_t;
	using InstanceIndex = std::uint32_t;

	class Scene {
	public:
		Scene() noexcept = default;
		virtual ~Scene() noexcept = default;

		[[nodiscard]] virtual InstanceIndex addMeshInstance(std::shared_ptr<Mesh> mesh) = 0;
		virtual void updateTransform(InstanceIndex instance, glm::fmat4x4 transform) = 0;
	};

	/**
	 * The abstracted renderer interface.
	 */
	class Renderer : public std::enable_shared_from_this<Renderer> {
	public:
		Renderer() noexcept = default;
		virtual ~Renderer() noexcept = default;

		[[nodiscard]] static std::shared_ptr<Renderer> createRenderer(GLFWwindow* window);

		[[nodiscard]] virtual std::unique_ptr<Buffer> createUniqueBuffer() = 0;
		[[nodiscard]] virtual std::shared_ptr<Buffer> createSharedBuffer() = 0;

		[[nodiscard]] virtual MaterialIndex getDefaultMaterialIndex() const noexcept {
			return 0;
		}
		[[nodiscard]] virtual MaterialIndex createMaterial(shaders::material_t material) = 0;

		[[nodiscard]] virtual std::shared_ptr<Mesh> createSharedMesh(
				std::span<const glm::fvec3> positions, std::span<const shaders::vertex_t> vertices,
				std::array<std::span<const glm::fvec2>, shaders::max_uv_sets> uvs, std::span<const index_t> indices,
				glm::fvec3 aabbCenter, glm::fvec3 aabbExtents,
				MaterialIndex materialIndex) = 0;

		[[nodiscard]] virtual std::shared_ptr<Scene> createSharedScene() = 0;

		[[nodiscard]] virtual std::shared_ptr<Image> createSharedImage(
			std::span<std::byte> imageData, glm::u32vec2 extents) = 0;

		[[nodiscard]] virtual std::shared_ptr<Sampler> getDefaultSampler() = 0;
		[[nodiscard]] virtual std::shared_ptr<Sampler> createSharedSampler(
			const fastgltf::Sampler& sampler) = 0; // TODO

		[[nodiscard]] virtual std::shared_ptr<Texture> createSharedTexture(
			std::shared_ptr<Image> image, std::shared_ptr<Sampler> sampler) = 0;

		/**
		 * If this returns false, the window might be minimised or being resized, forcing us to pause rendering shortly.
		 * In that case, glfwWaitEvents should be used.
		 */
		[[nodiscard]] virtual bool canRender() = 0;

		virtual void updateResolution(glm::u32vec2 resolution) = 0;
		/**
		 * Returns the resolution at which this renders the scene. Note that this might not be
		 * the same resolution the window has, since the renderer might use some upscaling
		 * technique.
		 */
		[[nodiscard]] virtual glm::u32vec2 getRenderResolution() const noexcept = 0;

		virtual void prepareFrame(std::size_t frameIndex) = 0;
		virtual bool draw(std::size_t frameIndex, Scene& scene,
						  const shaders::camera_t& camera, float dt) = 0;
	};
}
