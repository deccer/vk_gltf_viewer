#include <cassert>
#include <numbers>
#include <ranges>

#include <tracy/Tracy.hpp>

#include <fmt/xchar.h>
#include <fmt/std.h>

#include <vulkan/vk.hpp>
#include <vulkan/pipeline_builder.hpp>
#include <GLFW/glfw3.h> // After Vulkan includes so that it detects it
#include <imgui_impl_glfw.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <nvidia/dlss.hpp>
#if defined(VKV_NV_DLSS)
#include <nvsdk_ngx_helpers_vk.h>
#endif
#include <vk_gltf_viewer/scheduler.hpp>
#include <vk_gltf_viewer/application.hpp>
#include <vk_gltf_viewer/assets.hpp>
#include <spirv_manifest.hpp>

#include <visbuffer/visbuffer.h>

#include <fastgltf/tools.hpp>

namespace fg = fastgltf;

void glfwErrorCallback(int errorCode, const char* description) {
	if (errorCode != GLFW_NO_ERROR) {
		fmt::print(stderr, "GLFW error: 0x{:x} {}\n", errorCode, description);
	}
}

void glfwResizeCallback(GLFWwindow* window, int width, int height) {
	ZoneScoped;
	if (width > 0 && height > 0) {
		glm::u32vec2 res(width, height);
		decltype(auto) app = *static_cast<application*>(glfwGetWindowUserPointer(window));
		app.renderResolution = res;
		app.renderer->update_resolution(res);
	}
}

application::application(std::span<std::filesystem::path> gltfs) {
	ZoneScoped;

	// Initialize GLFW
	glfwSetErrorCallback(glfwErrorCallback);
	if (glfwInit() != GLFW_TRUE) {
		throw std::runtime_error("Failed to initialize GLFW");
	}
	deletion_queue.push([]() { glfwTerminate(); });

	// Get the main monitor's video mode
	auto* mainMonitor = glfwGetPrimaryMonitor();
	const auto* videoMode = glfwGetVideoMode(mainMonitor);

	// Create the window
	glfwDefaultWindowHints();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	window = glfwCreateWindow(
		static_cast<int>(static_cast<float>(videoMode->width) * 0.9f),
		static_cast<int>(static_cast<float>(videoMode->height) * 0.9f),
		"vk_viewer", nullptr, nullptr);
	if (window == nullptr)
		throw std::runtime_error("Failed to create window");
	deletion_queue.push([this]() { glfwDestroyWindow(window); });

	glfwSetWindowUserPointer(window, this);
	glfwSetWindowSizeCallback(window, glfwResizeCallback);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	//ImGui_ImplGlfw_InitForVulkan(window, true);
	ImGui_ImplGlfw_InitForOther(window, true);

	renderer = graphics::renderer::create_renderer(window);
	scene = renderer->create_shared_scene();
	camera = std::make_unique<camera_t>();

	for (auto& gltf : gltfs) {
		auto& task = asset_load_tasks.emplace_back(std::make_shared<asset_load_task>(renderer, gltf));
		task_scheduler.AddTaskSetToPipe(task.get());
	}
}

application::~application() noexcept {
	// TODO: is there perhaps some way to detach the tasks, so that we don't lock up
	//       the application during shutting down?
	for (auto& task : asset_load_tasks)
		task_scheduler.WaitforTask(task.get());
}

void application::update_render_resolution() {
	ZoneScoped;
	//auto swapchainExtent = toVector(swapchain->swapchain.extent);

#if defined(VKV_NV_DLSS)
	if (scalingMode == ResolutionScalingModes::DLSS) {
		auto settings = dlss::getRecommendedSettings(dlssQuality, swapchainExtent);
		renderResolution = settings.optimalRenderSize;

		device->timelineDeletionQueue->push([handle = dlssHandle]() {
			dlss::releaseFeature(handle);
		});
		dlssHandle = dlss::initFeature(*device, renderResolution, swapchainExtent);
	} else
#endif
	{
		//renderResolution = swapchainExtent;
	}

	firstFrame = true; // We need to re-transition the images since they've been recreated.
}

void application::add_asset_to_scene(asset_load_task& task) {
	ZoneScoped;
	textures.insert(textures.end(), task.textures.begin(), task.textures.end());

	fastgltf::iterateSceneNodes(*task.asset, 0, fastgltf::math::fmat4x4(),
								[&](fastgltf::Node& node, const fastgltf::math::fmat4x4& mat) {
		if (!node.meshIndex.has_value())
			return;

		auto& mesh = task.meshes[*node.meshIndex];
		for (auto& idx : mesh.primitive_indices) {
			auto instance = scene->add_mesh_instance(task.primitives[idx]);
			scene->update_transform(instance, glm::make_mat4x4(mat.data()));
		}
	});

	fmt::print("Finished loading asset: {}\n", task.asset_path);
    using namespace std::chrono_literals;
	//std::this_thread::sleep_for(2s);
}

void application::run() {
	ZoneScoped;

	std::size_t currentFrame = 0;
	while (!glfwWindowShouldClose(window)) {
		if (renderer->can_render()) {
			ZoneScopedN("glfwPollEvents");
			glfwPollEvents();
		} else {
			// This will wait until we get an event, like the resize event which will recreate the swapchain.
			glfwWaitEvents();
			continue;
		}

		// Check if any asset load task has completed, and get the Asset object
		for (auto& task : asset_load_tasks) {
			if (task->GetIsCompleteWithExceptions()) {
				add_asset_to_scene(*task);
				//vkQueueWaitIdle(device->graphicsQueue);
				//world->addAsset(task);
				task.reset();
			}
		}
		std::erase_if(asset_load_tasks, [](std::shared_ptr<asset_load_task>& value) {
			return !bool(value);
		});

		auto currentTime = glfwGetTime();
		deltaTime = currentTime - lastFrame;
		lastFrame = currentTime;

		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		render_ui();

		currentFrame = ++currentFrame % graphics::frame_overlap;

		renderer->prepare_frame(currentFrame);

		camera->update(window, deltaTime, renderer->get_render_resolution());

		renderer->draw(currentFrame, *scene, *camera, static_cast<float>(deltaTime));

		FrameMark;
		firstFrame = false;
	}
}

void application::render_ui() {
	ZoneScoped;
	if (ImGui::Begin("vk_gltf_viewer", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove)) {
		if (ImGui::BeginTabBar("#tabbar")) {
			if (ImGui::BeginTabItem("General")) {
				ImGui::SeparatorText("Performance");

				ImGui::Text("Frametime: %.2f ms", deltaTime * 1000.);
				ImGui::Text("FPS: %.2f", 1. / deltaTime);
				ImGui::Text("AFPS: %.2f rad/s", 2. * std::numbers::pi_v<double> / deltaTime); // Angular FPS

				ImGui::EndTabItem();
			}

			if (ImGui::BeginTabItem("Graphics")) {
				ImGui::SeparatorText("Resolution scaling");

				auto scalingModeName = std::find_if(availableScalingModes.begin(), availableScalingModes.end(), [&](auto& mode) {
					return mode.first == scalingMode;
				})->second;
				if (ImGui::BeginCombo("Resolution scaling", scalingModeName.data())) {
					for (const auto& [mode, name] : availableScalingModes) {
						bool isSelected = mode == scalingMode;
						if (ImGui::Selectable(name.data(), isSelected)) {
							scalingMode = mode;
							update_render_resolution();
						}
						if (isSelected)
							ImGui::SetItemDefaultFocus();
					}
					ImGui::EndCombo();
				}

#if defined(VKV_NV_DLSS)
				if (scalingMode == ResolutionScalingModes::DLSS) {
					auto dlssQualityName = std::find_if(dlss::modes.begin(), dlss::modes.end(), [&](auto& mode) {
						return mode.first == dlssQuality;
					})->second;
					if (ImGui::BeginCombo("DLSS Mode", dlssQualityName.data())) {
						for (const auto& [quality, name]: dlss::modes) {
							bool isSelected = quality == dlssQuality;
							if (ImGui::Selectable(name.data(), isSelected)) {
								dlssQuality = quality;
								updateRenderResolution();
							}
							if (isSelected)
								ImGui::SetItemDefaultFocus();
						}

						ImGui::EndCombo();
					}
				}
#endif

				ImGui::EndTabItem();
			}

			if (ImGui::BeginTabItem("Debug")) {
				ImGui::SeparatorText("Camera");
				auto pos = camera->get_position();
				ImGui::Text("Position: <%.2f, %.2f, %.2f>", pos.x, pos.y, pos.z);
				ImGui::DragFloat("Camera speed multiplier", &camera->speedMultiplier);

				ImGui::SeparatorText("Culling");
				ImGui::Checkbox("Freeze Camera frustum", &camera->freezeCameraFrustum);
				ImGui::Checkbox("Freeze Occlusion matrix", &camera->freezeCullingMatrix);
				//ImGui::Checkbox("Freeze animations", &world->freezeAnimations);

				ImGui::EndTabItem();
			}

			ImGui::EndTabBar();
		}
	}
	ImGui::End();

	// ImGui::ShowDemoWindow();

	ImGui::Render();
}
