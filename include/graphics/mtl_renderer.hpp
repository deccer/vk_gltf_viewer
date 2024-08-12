#pragma once

#include <exception>

#include <graphics/renderer.hpp>

#include <Foundation/NSSharedPtr.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLSampler.hpp>
#include <QuartzCore/CAMetalLayer.hpp>

#include <graphics/resource_table.hpp>
#include <graphics/imgui/mtl_renderer.hpp>

namespace graphics::metal {
class MtlRenderer;

class MtlBuffer : public graphics::Buffer {
	friend MtlRenderer;
	MTL::Buffer* buffer = nullptr;

public:
	explicit MtlBuffer() = default;
	~MtlBuffer() noexcept = default;
};

class MtlSampler final : public Sampler, public NS::SharedPtr<MTL::SamplerState> {
public:
	explicit MtlSampler(MTL::SamplerState* sampler)
			: NS::SharedPtr<MTL::SamplerState>(NS::TransferPtr(sampler)) {}
};

class MtlImage final : public Image, public NS::SharedPtr<MTL::Texture> {
public:
	explicit MtlImage(MTL::Texture* texture)
			: NS::SharedPtr<MTL::Texture>(NS::TransferPtr(texture)) {}
};

class MtlTexture final : public Texture {
	std::shared_ptr<MtlResourceTable> resourceTable;
	std::shared_ptr<MtlImage> image;
	std::shared_ptr<MtlSampler> sampler;

public:
	explicit MtlTexture(std::shared_ptr<MtlResourceTable> resourceTable, std::shared_ptr<MtlImage> image, std::shared_ptr<MtlSampler> sampler)
			: Texture(resourceTable->allocateSampledImage(*image.get(), *sampler.get())), resourceTable(std::move(resourceTable)), image(std::move(image)), sampler(std::move(sampler)) {}
	~MtlTexture() override {
		resourceTable->removeSampledImageHandle(getHandle());
	}
};

class MeshletMesh : public graphics::Mesh {
public:
	NS::SharedPtr<MTL::Buffer> vertexIndexBuffer;
	NS::SharedPtr<MTL::Buffer> primitiveIndexBuffer;
	NS::SharedPtr<MTL::Buffer> vertexBuffer;
	NS::SharedPtr<MTL::Buffer> meshletBuffer;

	glm::fvec3 aabbExtents;
	glm::fvec3 aabbCenter;

	std::uint32_t meshletCount;
	std::uint32_t materialIndex;

	explicit MeshletMesh() = default;
	~MeshletMesh() noexcept = default;
};

struct MeshletSceneMesh {
	std::shared_ptr<MeshletMesh> mesh;
	shaders::Primitive primitive;

	explicit MeshletSceneMesh(std::shared_ptr<MeshletMesh> mesh, shaders::Primitive primitive)
			: mesh(std::move(mesh)), primitive(primitive) {}
	MeshletSceneMesh(MeshletSceneMesh&& other) noexcept
			: mesh(std::move(other.mesh)), primitive(other.primitive) {}
};

struct MeshletDrawBuffers {
	NS::SharedPtr<MTL::Buffer> primitiveBuffer;
	NS::SharedPtr<MTL::Buffer> meshletDrawBuffer;
	NS::SharedPtr<MTL::Buffer> transformBuffer;
};

class MeshletScene : public graphics::Scene {
public:
	NS::SharedPtr<MTL::Device> device;

	std::vector<MeshletSceneMesh> meshes;
	std::vector<glm::fmat4x4> transforms;
	std::vector<shaders::MeshletDraw> meshletDraws;

	std::vector<MeshletDrawBuffers> drawBuffers;

	explicit MeshletScene(NS::SharedPtr<MTL::Device> device, std::size_t frameOverlap) : device(std::move(device)) {
		drawBuffers.resize(frameOverlap);
	}
	~MeshletScene() noexcept override = default;

	InstanceIndex addMeshInstance(std::shared_ptr<Mesh> mesh) override;
	void updateTransform(InstanceIndex instance, glm::fmat4x4 transform) override;

	void updateDrawBuffers(std::size_t frameIndex);
};

CA::MetalLayer* createMetalLayer(GLFWwindow* window);

struct VisbufferPass {
	NS::SharedPtr<MTL::RenderPipelineState> pipelineState;
	NS::SharedPtr<MTL::DepthStencilState> depthState;

	/** The pipeline used to draw AABBs to determine the  */
	NS::SharedPtr<MTL::RenderPipelineState> aabbPipelineState;

	NS::SharedPtr<MTL::Texture> visbuffer;
	NS::SharedPtr<MTL::Texture> depthTexture;
};

struct VisbufferResolvePass {
	NS::SharedPtr<MTL::ComputePipelineState> pipelineState;
};

class MtlRenderer : public graphics::Renderer {
	friend std::shared_ptr<Renderer> graphics::Renderer::createRenderer(GLFWwindow* window);

	/** This comes first so that it wraps around the entire lifetime of the renderer object */
	// NS::SharedPtr<NS::AutoreleasePool> pool;

	NS::SharedPtr<MTL::Device> device;

	CA::MetalLayer* layer;
	NS::SharedPtr<MTL::CommandQueue> commandQueue;
	dispatch_semaphore_t drawSemaphore;
	std::exception_ptr commandBufferException;

	std::shared_ptr<MtlResourceTable> resourceTable;

	NS::SharedPtr<MTL::Library> globalLibrary;

	std::unique_ptr<imgui::Renderer> imguiRenderer;

	std::vector<NS::SharedPtr<MTL::Buffer>> cameraBuffers;

	std::vector<shaders::Material> materials;
	std::vector<NS::SharedPtr<MTL::Buffer>> materialBuffers;

	std::shared_ptr<MtlSampler> defaultSampler;

	VisbufferPass visbufferPass;
	VisbufferResolvePass visbufferResolvePass;

	void initVisbufferPass();
	void initVisbufferResolvePass();

public:
	explicit MtlRenderer(GLFWwindow* window);
	~MtlRenderer() noexcept override;

	std::unique_ptr<Buffer> createUniqueBuffer() override;
	std::shared_ptr<Buffer> createSharedBuffer() override;

	MaterialIndex createMaterial(shaders::Material material) override;

	std::shared_ptr<Mesh> createSharedMesh(
			std::span<shaders::Vertex> vertexBuffer, std::span<index_t> indexBuffer,
			glm::fvec3 aabbCenter, glm::fvec3 aabbExtents,
			MaterialIndex materialIndex) override;

	std::shared_ptr<Scene> createSharedScene() override;

	std::shared_ptr<Image> createSharedImage(
		std::span<std::byte> imageData, glm::u32vec2 extents) override;

	std::shared_ptr<Sampler> getDefaultSampler() override;
	std::shared_ptr<Sampler> createSharedSampler(
		const fastgltf::Sampler& sampler) override;
	std::shared_ptr<Texture> createSharedTexture(
		std::shared_ptr<Image> image, std::shared_ptr<Sampler> sampler) override;

	bool canRender() override {
		return true; // TODO: Detect window being minimized or sth
	}

	void updateResolution(glm::u32vec2 resolution) override;
	auto getRenderResolution() const noexcept -> glm::u32vec2 override;

	void prepareFrame(std::size_t frameIndex) override;
	bool draw(std::size_t frameIndex, Scene& world,
			  const shaders::Camera& camera, float dt) override;
};
}
