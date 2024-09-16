#pragma once

#include <deque>
#include <filesystem>
#include <span>
#include <vector>

#include <GLFW/glfw3.h>

#include <vulkan/vk.hpp>
#include <vulkan/command_pool.hpp>

#if defined(VKV_NV_DLSS)
#include <nvsdk_ngx_defs.h>
#endif

#include <graphics/renderer.hpp>

#include <vk_gltf_viewer/device.hpp>
#include <vk_gltf_viewer/swapchain.hpp>
#include <vk_gltf_viewer/deletion_queue.hpp>
#include <vk_gltf_viewer/image.hpp>
#include <vk_gltf_viewer/assets.hpp>
#include <vk_gltf_viewer/camera.hpp>

#include <entt/entt.hpp>

enum class ResolutionScalingModes {
	None,
#if defined(VKV_NV_DLSS)
	DLSS,
#endif
};

struct node_component_t {
	entt::entity parent = entt::null;

	entt::entity first_child = entt::null;

	// prev/next values used as iterators for iterating through a list of
	// children from the parent node
	entt::entity prev = entt::null;
	entt::entity next = entt::null;
};

struct name_component_t {
	std::string name;
};

struct transform_component_t {
	glm::fvec3 position = glm::fvec3(0.f);
	glm::fquat rotation = glm::fquat::wxyz(1.f, 0.f, 0.f, 0.f);
	glm::fvec3 scale = glm::fvec3(1.f);

	glm::fmat4x4 transform = glm::fmat4x4(1.f);
};

struct dirty_transform_tag_t {};

struct mesh_component_t {
	std::shared_ptr<graphics::mesh_t> mesh;
	graphics::instance_index instance_index;
};

/** The main Application class */
class application {
	friend void glfwResizeCallback(GLFWwindow* window, int width, int height);

	std::vector<std::shared_ptr<asset_load_task>> asset_load_tasks;

	/** The global deletion queue for all sorts of objects */
	DeletionQueue deletion_queue;

	GLFWwindow* window;
	std::shared_ptr<graphics::renderer> renderer;

	entt::registry registry;
	entt::entity root_scene_node = entt::null;
	std::shared_ptr<graphics::scene_t> scene;
	std::unique_ptr<camera_t> camera;

	entt::entity selected_node = entt::null;

	std::vector<std::shared_ptr<graphics::texture_t>> textures;

	graphics::scaling_modes_e scaling_mode = graphics::scaling_modes_e::none;
	std::size_t scaling_mode_preset = 0;

	bool firstFrame = true;
	double deltaTime = 0., lastFrame = 0.;

	void update_render_resolution();
	void add_asset_to_scene(asset_load_task& task);
	void render_ui();
	void render_scene_tree(entt::entity node);

public:
	explicit application(std::span<std::filesystem::path> gltfs);
	~application() noexcept;

	void run();
};
