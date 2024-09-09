#pragma once

#include <memory>

#include <glm/vec2.hpp>

#include <vulkan/vk.hpp>
#include <vulkan/command_pool.hpp>
#include <vulkan/sync_pools.hpp>

#if defined(VKV_NV_DLSS)
#include <nvsdk_ngx_defs.h>
#endif

#include <vk_gltf_viewer/device.hpp>
#include <vk_gltf_viewer/swapchain.hpp>

#include <graphics/imgui/vk_renderer.hpp>
#include <graphics/renderer.hpp>

namespace graphics::vulkan {
struct VkMesh : graphics::mesh_t {
	/** The buffer handles corresponding to the buffers in each shaders::Primitive. */
	std::unique_ptr<ScopedBuffer> vertexIndexBuffer;
	std::unique_ptr<ScopedBuffer> primitiveIndexBuffer;
	std::unique_ptr<ScopedBuffer> vertexBuffer;
	std::unique_ptr<ScopedBuffer> meshletBuffer;

	glm::fvec3 aabbExtents;
	glm::fvec3 aabbCenter;

	std::uint32_t meshletCount;
	std::uint32_t materialIndex;
};

struct DrawBuffers {
	bool isMeshletBufferBuilt = false;
	std::unique_ptr<ScopedBuffer> meshletDrawBuffer;
	std::unique_ptr<ScopedBuffer> transformBuffer;
};

class VkScene : graphics::scene_t {
	std::vector<std::shared_ptr<mesh_t>> meshes;
	std::unique_ptr<ScopedBuffer> primitiveBuffer;
	std::unique_ptr<ScopedBuffer> materialBuffer;

	std::vector<DrawBuffers> drawBuffers;

	void rebuildDrawBuffer(std::size_t frameIndex);
	void updateTransformBuffer(std::size_t frameIndex);

public:
	instance_index add_mesh_instance(std::shared_ptr<mesh_t> mesh) override;
	void update_transform(instance_index instance, glm::fmat4x4 transform) override;

	void updateDrawBuffers(std::size_t frameIndex, float dt);
};

/** Sync primitives required for frame synchronization around presenting and work submission */
struct FrameSyncData {
	std::unique_ptr<vk::Semaphore> imageAvailable;
	std::unique_ptr<vk::Semaphore> renderingFinished;
	std::unique_ptr<vk::Fence> presentFinished;
};

/** A CommandPool for each frame, together with pre-allocated command buffers */
struct FrameCommandPool {
	vk::CommandPool commandPool;
	VkCommandBuffer commandBuffer;
};

enum class ResolutionScalingModes {
	None,
#if defined(VKV_NV_DLSS)
	DLSS,
#endif
};

class VkRenderer : public graphics::renderer {
	friend std::shared_ptr<renderer> graphics::renderer::create_renderer(GLFWwindow* window);

	std::unique_ptr<Instance> instance;
	std::unique_ptr<Device> device;
	std::unique_ptr<Swapchain> swapchain;

	std::unique_ptr<imgui::Renderer> imguiRenderer;

	glm::u32vec2 renderResolution;
	ResolutionScalingModes scalingMode = ResolutionScalingModes::None;
	std::vector<std::pair<ResolutionScalingModes, std::string_view>> availableScalingModes;

#if defined(VKV_NV_DLSS)
	NVSDK_NGX_Handle* dlssHandle = nullptr;
	NVSDK_NGX_PerfQuality_Value dlssQuality = NVSDK_NGX_PerfQuality_Value_Balanced;
#endif

	std::vector<FrameSyncData> frameSyncData;
	std::vector<FrameCommandPool> frameCommandPools;

	bool swapchainNeedsRebuild = false;

public:
	std::unique_ptr<buffer_t> create_unique_buffer() override;
	std::shared_ptr<buffer_t> create_shared_buffer() override;

	std::shared_ptr<mesh_t> create_shared_mesh(
			std::span<const glm::fvec3> positions, std::span<const shaders::vertex_t> vertices,
			std::array<std::span<const glm::fvec2>, shaders::max_uv_sets> uvs, std::span<const index_t> indices,
			glm::fvec3 aabbCenter, glm::fvec3 aabbExtents,
			material_index materialIndex) override;

	bool can_render() override {
		return !swapchainNeedsRebuild;
	}

	void update_resolution(glm::u32vec2 resolution) override;

	void prepare_frame(std::size_t frameIndex) override;
	bool draw(std::size_t frameIndex, scene_t& world, const shaders::camera_t& camera, float dt) override;
};
} // namespace graphics::vulkan
