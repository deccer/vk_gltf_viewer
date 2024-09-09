#include <tracy/Tracy.hpp>

#include <meshoptimizer.h>

#include <Metal/MTLLibrary.hpp>
#include <Metal/MTLCommandBuffer.hpp>
#include <Metal/MTLCommandQueue.hpp>
#include <Metal/MTLRenderPipeline.hpp>
#include <Metal/MTLDepthStencil.hpp>

#include <util.hpp>
#include <graphics/mtl_renderer.hpp>

#include <fastgltf/util.hpp>
#include <Metal/MTLBlitCommandEncoder.hpp>
#include <Metal/MTLComputeCommandEncoder.hpp>
#include <Metal/MTLComputePipeline.hpp>
#include <Metal/MTLPipeline.hpp>
#include <Metal/MTLSampler.hpp>

#include "visbuffer/visbuffer.h"

namespace gmtl = graphics::metal;

graphics::instance_index gmtl::meshlet_scene::add_mesh_instance(std::shared_ptr<mesh_t> mesh) {
	ZoneScoped;
	auto meshletMesh = std::static_pointer_cast<meshlet_mesh>(mesh);
	//auto meshletMesh = std::dynamic_pointer_cast<MeshletMesh>(mesh);

	// Add to mesh list if it's not there already
	std::optional<std::uint32_t> primitive_index;
	{
		auto it = meshes.begin();
		while (it != meshes.end()) {
			if (it->mesh == meshletMesh) {
				primitive_index = std::distance(meshes.begin(), it);
				break;
			}
			++it;
		}
		if (it == meshes.end()) {
			primitive_index = meshes.size();
			meshes.emplace_back(meshletMesh, shaders::primitive_t {
				.vertex_index_buffer = meshletMesh->vertex_index_buffer->gpuAddress(),
				.primitive_index_buffer = meshletMesh->primitive_index_buffer->gpuAddress(),
				.position_buffer = meshletMesh->position_buffer->gpuAddress(),
				.vertex_buffer = meshletMesh->vertex_buffer->gpuAddress(),
				.meshlet_buffer = meshletMesh->meshlet_buffer->gpuAddress(),
				.uv_buffers = transform_array(meshletMesh->uv_buffers, [](auto& buffer) { return buffer->gpuAddress(); }),
				.aabb_extents = meshletMesh->aabb_extents,
				.aabb_center = meshletMesh->aabb_extents,
				.meshlet_count = meshletMesh->meshlet_count,
				.material_index = meshletMesh->material_index,
			});
		}
	}

	assert(primitive_index.has_value()); // Just to shut up some static analyzers.

	auto instance_index = static_cast<std::uint32_t>(transforms.size());

	for (std::uint32_t i = 0; i < meshletMesh->meshlet_count; ++i) {
		meshlet_draws.emplace_back(shaders::meshlet_draw_t {
			.primitive_index = *primitive_index,
			.meshlet_index = i,
			.transform_index = instance_index,
		});
	}

	transforms.emplace_back();
	return instance_index;
}

void gmtl::meshlet_scene::update_transform(graphics::instance_index instance, glm::fmat4x4 transform) {
	ZoneScoped;
	transforms[instance] = {
		.matrix = transform,
		.inverse_matrix = glm::inverse(transform),
	};
}

void gmtl::meshlet_scene::update_draw_buffers(std::size_t frameIndex) {
	ZoneScoped;
	auto& draw_buffer = draw_buffers[frameIndex];

	{
		auto length = draw_buffer.meshlet_draw_buffer ? draw_buffer.meshlet_draw_buffer->length() : 0;
		auto requiredLength = meshlet_draws.size() * sizeof(decltype(meshlet_draws)::value_type);
		if (requiredLength > length) {
			draw_buffer.meshlet_draw_buffer = NS::TransferPtr(
					device->newBuffer(requiredLength, MTL::ResourceStorageModeShared));
			draw_buffer.meshlet_draw_buffer->setLabel(MTLSTR("Meshlet draw buffer"));
		}
		std::memcpy(draw_buffer.meshlet_draw_buffer->contents(), meshlet_draws.data(), requiredLength);
	}

	{
		auto length = draw_buffer.transform_buffer ? draw_buffer.transform_buffer->length() : 0;
		auto requiredLength = transforms.size() * sizeof(decltype(transforms)::value_type);
		if (requiredLength > length) {
			draw_buffer.transform_buffer = NS::TransferPtr(
					device->newBuffer(requiredLength, MTL::ResourceStorageModeShared));
			draw_buffer.transform_buffer->setLabel(MTLSTR("Transform buffer"));
		}
		std::memcpy(draw_buffer.transform_buffer->contents(), transforms.data(), requiredLength);
	}

	{
		auto length = draw_buffer.primitive_buffer ? draw_buffer.primitive_buffer->length() : 0;
		auto requiredLength = meshes.size() * sizeof(decltype(meshes)::value_type);
		if (requiredLength > length) {
			draw_buffer.primitive_buffer = NS::TransferPtr(
					device->newBuffer(requiredLength, MTL::ResourceStorageModeShared));
			draw_buffer.primitive_buffer->setLabel(MTLSTR("Primitive buffer"));
		}
		for (std::size_t i = 0; auto& mesh : meshes)
			static_cast<shaders::primitive_t*>(draw_buffer.primitive_buffer->contents())[i++]
				= mesh.primitive;
	}
}

static constexpr std::array<MTL::PixelFormat, 3> gbuffer_formats {{
	MTL::PixelFormatRGBA8Unorm, // Color
#if GBUFFER_NORMAL_ENCODING == 1
	MTL::PixelFormatRG16Snorm, // Normals
#else
	MTL::PixelFormatRGBA16Snorm,
#endif
	MTL::PixelFormatRG16Float, // Metallic & Roughness
}};

void gmtl::visbuffer_pass::init_pass(meshlet_renderer& renderer) {
	ZoneScoped;
	auto device = renderer.device;

	auto* objectFunction = renderer.global_library->newFunction(MTLSTR("visbuffer_object"))->autorelease();
	auto* meshFunction = renderer.global_library->newFunction(MTLSTR("visbuffer_mesh"))->autorelease();
	auto* fragFunction = renderer.global_library->newFunction(MTLSTR("visbuffer_frag"))->autorelease();

	auto* visbuffer_pipeline_desc = MTL::MeshRenderPipelineDescriptor::alloc()->init()->autorelease();
	visbuffer_pipeline_desc->setObjectFunction(objectFunction);
	visbuffer_pipeline_desc->setMeshFunction(meshFunction);
	visbuffer_pipeline_desc->setFragmentFunction(fragFunction);
	visbuffer_pipeline_desc->setRasterSampleCount(1);

	for (std::size_t i = 0; auto& format : gbuffer_formats)
		visbuffer_pipeline_desc->colorAttachments()->object(i++)->setPixelFormat(format);
	visbuffer_pipeline_desc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);

	NS::Error* error = nullptr;
	visbuffer_pipeline = NS::TransferPtr(
			device->newRenderPipelineState(visbuffer_pipeline_desc, MTL::PipelineOptionNone, nullptr, &error));
	if (!visbuffer_pipeline) {
		fmt::print("{}", error->localizedDescription()->utf8String());
	}

	auto* gbuffer_tile_function = renderer.global_library->newFunction(MTLSTR("gbuffer_generation"));
	auto* gbuffer_tile_desc = MTL::TileRenderPipelineDescriptor::alloc()->init()->autorelease();
	gbuffer_tile_desc->setTileFunction(gbuffer_tile_function);

	for (std::size_t i = 0; auto& format : gbuffer_formats)
		gbuffer_tile_desc->colorAttachments()->object(i++)->setPixelFormat(format);

	gbuffer_tile_pipeline = NS::TransferPtr(
		device->newRenderPipelineState(gbuffer_tile_desc, MTL::PipelineOptionNone, nullptr, &error));
	if (!gbuffer_tile_pipeline) {
		fmt::print("{}", error->localizedDescription()->utf8String());
	}

	// Create the depth state with reverse depth
	auto* depthStateDesc = MTL::DepthStencilDescriptor::alloc()->init();
	depthStateDesc->setDepthWriteEnabled(true);
	depthStateDesc->setDepthCompareFunction(MTL::CompareFunctionGreaterEqual);

	depth_state = NS::TransferPtr(device->newDepthStencilState(depthStateDesc));

	update_resolution(renderer);
}

void gmtl::visbuffer_pass::update_resolution(meshlet_renderer& renderer) {
	ZoneScoped;
	auto device = renderer.device;

	auto size = renderer.layer->drawableSize();

	auto* albedo_desc = MTL::TextureDescriptor::texture2DDescriptor(
		gbuffer_formats[0], size.width, size.height, false);
	albedo_desc->setUsage(MTL::TextureUsageRenderTarget);
	albedo_desc->setStorageMode(MTL::StorageModePrivate);

	albedo_texture = NS::TransferPtr(device->newTexture(albedo_desc));
	albedo_texture->setLabel(MTLSTR("Albedo"));

	auto* normal_desc = MTL::TextureDescriptor::texture2DDescriptor(
		gbuffer_formats[1], size.width, size.height, false);
	normal_desc->setUsage(MTL::TextureUsageRenderTarget);
	normal_desc->setStorageMode(MTL::StorageModePrivate);

	normal_texture = NS::TransferPtr(device->newTexture(normal_desc));
	normal_texture->setLabel(MTLSTR("Normals"));

	auto* mr_desc = MTL::TextureDescriptor::texture2DDescriptor(
		gbuffer_formats[2], size.width, size.height, false);
	mr_desc->setUsage(MTL::TextureUsageRenderTarget);
	mr_desc->setStorageMode(MTL::StorageModePrivate);

	metallic_roughness_texture = NS::TransferPtr(device->newTexture(mr_desc));
	metallic_roughness_texture->setLabel(MTLSTR("Metallic roughness"));

	auto* depthDesc = MTL::TextureDescriptor::texture2DDescriptor(
			MTL::PixelFormatDepth32Float, size.width, size.height, false);
	depthDesc->setUsage(MTL::TextureUsageShaderRead & MTL::TextureUsageRenderTarget);
	depthDesc->setStorageMode(MTL::StorageModePrivate);

	depth_texture = NS::TransferPtr(device->newTexture(depthDesc));
	depth_texture->setLabel(MTLSTR("Depth texture"));
}

void gmtl::shading_pass::init_pass(meshlet_renderer& renderer) {
	auto& device = renderer.device;

	auto* gbuffer_tile_function = renderer.global_library->newFunction(MTLSTR("gbuffer_shading"));
	auto* gbuffer_tile_desc = MTL::TileRenderPipelineDescriptor::alloc()->init()->autorelease();
	gbuffer_tile_desc->setTileFunction(gbuffer_tile_function);

	gbuffer_tile_desc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA16Float);
	for (std::size_t i = 1; auto& format : gbuffer_formats)
		gbuffer_tile_desc->colorAttachments()->object(i++)->setPixelFormat(format);

	NS::Error* error = nullptr;
	shading_pass_pipeline = NS::TransferPtr(
		device->newRenderPipelineState(gbuffer_tile_desc, MTL::PipelineOptionNone, nullptr, &error));
	if (!shading_pass_pipeline) {
		fmt::print("{}", error->localizedDescription()->utf8String());
	}

	update_resolution(renderer);
}

void gmtl::shading_pass::update_resolution(meshlet_renderer& renderer) {
	ZoneScoped;
	auto& device = renderer.device;
}

gmtl::meshlet_renderer::meshlet_renderer(GLFWwindow* window) {
	ZoneScoped;
	// pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

	device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());

	layer = create_metal_layer(window);
	layer->setDevice(device.get());
	layer->setPixelFormat(MTL::PixelFormatRGBA16Float);

	command_queue = NS::TransferPtr(device->newCommandQueue());

	draw_semaphore = dispatch_semaphore_create(frame_overlap);

	resource_table = std::make_shared<mtl_resource_table>(device);

	auto* libraryUrl = NS::URL::alloc()->initFileURLWithPath(MTLSTR("shaders.metallib"))->autorelease();

	NS::Error* error = nullptr;
	global_library = NS::TransferPtr(device->newLibrary(libraryUrl, &error));
	if (error) {
		fmt::print("{}", error->localizedDescription()->utf8String());
	}

	imgui_renderer = std::make_unique<imgui::imgui_renderer>(
			device, global_library, resource_table, layer->pixelFormat());

	camera_buffers.resize(frame_overlap);
	for (std::size_t i = 0; auto& camera : camera_buffers) {
		camera = NS::TransferPtr(device->newBuffer(sizeof(shaders::camera_t), MTL::StorageModeShared));
		auto str = fmt::format("Camera buffer {}", i++);
		camera->setLabel(NS::String::string(str.c_str(), NS::UTF8StringEncoding));
	}

	// Create a default material with glTF defaults.
	materials.emplace_back(shaders::material_t {
		.albedo_factor = glm::fvec4(1.f),
		.albedo = {
			.index = shaders::invalid_handle,
			.uv_set = 0,
		},
		.metallic_factor = 1.f,
		.roughness_factor = 1.f,
		.metallic_roughness = {
			.index = shaders::invalid_handle,
			.uv_set = 0,
		},
		.alpha_cutoff = 0.5f,
		.double_sided = false,
	});
	material_buffers.resize(frame_overlap);

	// Create the default sampler.
	// When texture.sampler is undefined, a sampler with repeat wrapping (in both directions) and auto filtering MUST be used.
	auto* samplerDesc = MTL::SamplerDescriptor::alloc()->init()->autorelease();
	samplerDesc->setRAddressMode(MTL::SamplerAddressModeRepeat);
	samplerDesc->setSAddressMode(MTL::SamplerAddressModeRepeat);
	default_sampler = std::make_shared<mtl_sampler>(device->newSamplerState(samplerDesc));

	visbuffer_pass.init_pass(*this);
	shading_pass.init_pass(*this);
}

gmtl::meshlet_renderer::~meshlet_renderer() noexcept = default;

std::unique_ptr<graphics::buffer_t> gmtl::meshlet_renderer::create_unique_buffer() {
	ZoneScoped;
	return std::make_unique<mtl_buffer>();
}

std::shared_ptr<graphics::buffer_t> gmtl::meshlet_renderer::create_shared_buffer() {
	ZoneScoped;
	return std::make_shared<mtl_buffer>();
}

graphics::material_index gmtl::meshlet_renderer::create_material(shaders::material_t material) {
	ZoneScoped;
	assert(materials.size() < std::numeric_limits<material_index>::max());
	const auto index = materials.size();
	materials.emplace_back(material);
	return static_cast<material_index>(index);
}

std::shared_ptr<graphics::mesh_t> gmtl::meshlet_renderer::create_shared_mesh(
				std::span<const glm::fvec3> positions, std::span<const shaders::vertex_t> vertices,
				std::array<std::span<const glm::fvec2>, shaders::max_uv_sets> uvs, std::span<const index_t> indices,
		glm::fvec3 aabbCenter, glm::fvec3 aabbExtents, material_index materialIndex) {
	ZoneScoped;
	static constexpr auto coneWeight = 0.f; // We leave this as 0 because we're not using cluster cone culling.
	static constexpr auto maxPrimitives = fastgltf::alignDown(shaders::maxPrimitives, 4U); // meshopt requires the primitive count to be aligned to 4.
	std::size_t maxMeshlets = meshopt_buildMeshletsBound(indices.size(), shaders::maxVertices, maxPrimitives);

	std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
	std::vector<std::uint32_t> meshletVertices(maxMeshlets * shaders::maxVertices);
	std::vector<std::uint8_t> meshletTriangles(maxMeshlets * maxPrimitives * 3);

	auto mesh = std::make_shared<meshlet_mesh>();
	mesh->aabb_center = aabbCenter;
	mesh->aabb_extents = aabbExtents;
	mesh->material_index = materialIndex;

	// Create position buffer
	mesh->position_buffer = NS::TransferPtr(device->newBuffer(positions.size_bytes(), MTL::ResourceStorageModeShared));
	std::memcpy(mesh->position_buffer->contents(), positions.data(), positions.size_bytes());

	// Create vertex buffer
	mesh->vertex_buffer = NS::TransferPtr(device->newBuffer(vertices.size_bytes(), MTL::ResourceStorageModeShared));
	std::memcpy(mesh->vertex_buffer->contents(), vertices.data(), vertices.size_bytes());

	// Create UV buffers
	for (std::size_t i = 0; auto& uv_data : uvs) {
		if (uv_data.empty())
			continue;
		mesh->uv_buffers[i] = NS::TransferPtr(device->newBuffer(uv_data.size_bytes(), MTL::ResourceStorageModeShared));
		std::memcpy(mesh->uv_buffers[i]->contents(), uv_data.data(), uv_data.size_bytes());
		++i;
	}

	// Build meshlets & trim buffers accordingly
	{
		mesh->meshlet_count = meshopt_buildMeshlets(
			meshlets.data(), meshletVertices.data(), meshletTriangles.data(),
			indices.data(), indices.size(),
			&positions[0].x, positions.size(), sizeof(decltype(positions)::value_type),
			shaders::maxVertices, maxPrimitives, coneWeight);

		const auto& lastMeshlet = meshlets[mesh->meshlet_count - 1];
		meshletVertices.resize(lastMeshlet.vertex_count + lastMeshlet.vertex_offset);
		meshletTriangles.resize(((lastMeshlet.triangle_count * 3 + 3) & ~3) + lastMeshlet.triangle_offset);
		meshlets.resize(mesh->meshlet_count);
	}

	// Create meshlet buffer & transform meshlets
	mesh->meshlet_buffer = NS::TransferPtr(device->newBuffer(meshlets.size() * sizeof(shaders::meshlet_t), MTL::ResourceStorageModeShared));
	auto* glslMeshlets = static_cast<shaders::meshlet_t*>(mesh->meshlet_buffer->contents());
	for (std::size_t i = 0; auto& meshlet : meshlets) {
		meshopt_optimizeMeshlet(&meshletVertices[meshlet.vertex_offset],
								&meshletTriangles[meshlet.triangle_offset],
								meshlet.triangle_count, meshlet.vertex_count);

		// Compute meshlet bounds
		auto& initialVertex = positions[meshletVertices[meshlet.vertex_offset]];
		auto min = glm::vec3(initialVertex), max = glm::vec3(initialVertex);

		for (std::size_t j = 1; j < meshlet.vertex_count; ++j) {
			std::uint32_t vertexIndex = meshletVertices[meshlet.vertex_offset + j];
			auto& vertex = positions[vertexIndex];

			// The glm::min and glm::max functions are all component-wise.
			min = glm::min(min, vertex);
			max = glm::max(max, vertex);
		}

		// We can convert the count variables to a uint8_t since shaders::maxVertices and shaders::maxPrimitives both fit in 8-bits.
		assert(meshlet.vertex_count <= std::numeric_limits<std::uint8_t>::max());
		assert(meshlet.triangle_count <= std::numeric_limits<std::uint8_t>::max());
		auto center = (min + max) * 0.5f;
		glslMeshlets[i++] = shaders::meshlet_t {
			.vertexOffset = meshlet.vertex_offset,
			.triangleOffset = meshlet.triangle_offset,
			.vertexCount = static_cast<std::uint8_t>(meshlet.vertex_count),
			.triangleCount = static_cast<std::uint8_t>(meshlet.triangle_count),
			.aabb_extents = max - center,
			.aabb_center = center,
		};
	}

	// Finally, copy the index buffers
	mesh->vertex_index_buffer = NS::TransferPtr(device->newBuffer(meshletVertices.size() * sizeof(decltype(meshletVertices)::value_type), MTL::StorageModeShared));
	std::memcpy(mesh->vertex_index_buffer->contents(), meshletVertices.data(), mesh->vertex_index_buffer->length());

	mesh->primitive_index_buffer = NS::TransferPtr(device->newBuffer(meshletTriangles.size() * sizeof(decltype(meshletTriangles)::value_type), MTL::StorageModeShared));
	std::memcpy(mesh->primitive_index_buffer->contents(), meshletTriangles.data(), mesh->primitive_index_buffer->length());

	return mesh;
}

std::shared_ptr<graphics::scene_t> gmtl::meshlet_renderer::create_shared_scene() {
	ZoneScoped;
	return std::make_shared<meshlet_scene>(device, graphics::frame_overlap);
}

std::shared_ptr<graphics::sampler_t> gmtl::meshlet_renderer::get_default_sampler() {
	return default_sampler;
}

MTL::SamplerAddressMode get_address_mode(fastgltf::Wrap wrap) {
	switch (wrap) {
		using enum fastgltf::Wrap;
		case ClampToEdge:
			return MTL::SamplerAddressModeClampToEdge;
		case MirroredRepeat:
			return MTL::SamplerAddressModeMirrorRepeat;
		case Repeat:
		default:
			return MTL::SamplerAddressModeRepeat;
	}
}

MTL::SamplerMinMagFilter get_filter(fastgltf::Filter filter) {
	switch (filter) {
		using enum fastgltf::Filter;
		case Nearest:
		case NearestMipMapNearest:
		case NearestMipMapLinear:
		default:
			return MTL::SamplerMinMagFilterNearest;

		case Linear:
		case LinearMipMapNearest:
		case LinearMipMapLinear:
			return MTL::SamplerMinMagFilterLinear;
	}
}

MTL::SamplerMipFilter get_mip_filter(fastgltf::Filter filter) {
	switch (filter) {
		using enum fastgltf::Filter;
		case NearestMipMapNearest:
		case LinearMipMapNearest:
			return MTL::SamplerMipFilterNearest;

		case NearestMipMapLinear:
		case LinearMipMapLinear:
		default:
			return MTL::SamplerMipFilterLinear;
	}
}

std::shared_ptr<graphics::sampler_t> gmtl::meshlet_renderer::create_shared_sampler(
		const fastgltf::Sampler& sampler) {
	auto* sampler_desc = MTL::SamplerDescriptor::alloc()->init()->autorelease();
	sampler_desc->setSAddressMode(get_address_mode(sampler.wrapS));
	sampler_desc->setTAddressMode(get_address_mode(sampler.wrapT));
	sampler_desc->setMinFilter(get_filter(sampler.minFilter.value_or(fastgltf::Filter::Nearest)));
	sampler_desc->setMagFilter(get_filter(sampler.magFilter.value_or(fastgltf::Filter::Nearest)));
	sampler_desc->setMipFilter(get_mip_filter(sampler.minFilter.value_or(fastgltf::Filter::Nearest)));
	return std::make_shared<mtl_sampler>(device->newSamplerState(sampler_desc));
}

std::shared_ptr<graphics::image_t> gmtl::meshlet_renderer::create_shared_image(
	std::span<std::byte> imageData, glm::u32vec2 extents) {
ZoneScoped;
	auto* descriptor = MTL::TextureDescriptor::alloc()->init()->autorelease();
	descriptor->setStorageMode(MTL::StorageModeShared);
	descriptor->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
	descriptor->setUsage(MTL::TextureUsageShaderRead);
	descriptor->setAllowGPUOptimizedContents(true);
	descriptor->setWidth(extents.x);
	descriptor->setHeight(extents.y);
	descriptor->setDepth(1);

	auto mipmap_count = static_cast<NS::UInteger>(
		ceil(log2(fastgltf::max(extents.x, extents.y))));
	descriptor->setMipmapLevelCount(fastgltf::max<NS::UInteger>(mipmap_count, 1U));

	auto* texture = device->newTexture(descriptor);

	texture->replaceRegion(
		MTL::Region::Make2D(0, 0, extents.x, extents.y),
		0, imageData.data(), sizeof(std::uint32_t) * extents.x);

	if (mipmap_count > 1) {
		// TODO: (1) Expose API that takes mipmap data, if it already exists from the image file.
		//       (2) Avoid waiting on the command buffer here, and instead only wait for it completing
		//           on our main command buffer, just before we actually use the textures.
		auto* buffer = command_queue->commandBuffer();
		auto* blit_encoder = buffer->blitCommandEncoder();
		blit_encoder->generateMipmaps(texture);
		blit_encoder->endEncoding();
		buffer->commit();
		buffer->waitUntilCompleted();
	}

	return std::make_shared<mtl_image>(texture);
}

std::shared_ptr<graphics::texture_t> gmtl::meshlet_renderer::create_shared_texture(std::shared_ptr<image_t> image, std::shared_ptr<sampler_t> sampler) {
	auto mtlImage = std::static_pointer_cast<mtl_image>(image);
	auto mtlSampler = std::static_pointer_cast<mtl_sampler>(sampler);
	return std::make_shared<mtl_texture>(
		resource_table,
		mtlImage,
		mtlSampler);
}

void gmtl::meshlet_renderer::update_resolution(glm::u32vec2 resolution) {
	ZoneScoped;
	resolution *= 2; // TODO: Get CA::Layer::contentScale here.
	layer->setDrawableSize(CGSizeMake(resolution.x, resolution.y));

	visbuffer_pass.update_resolution(*this);
	shading_pass.update_resolution(*this);
}

auto gmtl::meshlet_renderer::get_render_resolution() const noexcept -> glm::u32vec2 {
	auto drawableSize = layer->drawableSize();
	return { drawableSize.width, drawableSize.height };
}

void gmtl::meshlet_renderer::prepare_frame(std::size_t frameIndex) {
	ZoneScoped;
	dispatch_semaphore_wait(draw_semaphore, DISPATCH_TIME_FOREVER);
	if (command_buffer_exception) {
		std::rethrow_exception(command_buffer_exception);
	}

	auto& material_buffer = material_buffers[frameIndex];
	if (const auto requiredSize = materials.size() * sizeof(shaders::material_t);
		!material_buffer || material_buffer->length() < requiredSize) {
		material_buffer = NS::TransferPtr(device->newBuffer(requiredSize, MTL::StorageModeShared));
		material_buffer->setLabel(MTLSTR("Material buffer"));
		std::memcpy(material_buffer->contents(), materials.data(), requiredSize);
	}
}

bool gmtl::meshlet_renderer::draw(std::size_t frameIndex, graphics::scene_t& gworld,
							 const shaders::camera_t& camera, float dt) {
	ZoneScoped;
	auto* pool = NS::AutoreleasePool::alloc()->init();

	auto& scene = dynamic_cast<meshlet_scene&>(gworld);

	auto* drawable = layer->nextDrawable();

	scene.update_draw_buffers(frameIndex);
	*static_cast<shaders::camera_t*>(camera_buffers[frameIndex]->contents()) = camera;

	auto* buffer_desc = MTL::CommandBufferDescriptor::alloc()->init()->autorelease();
	buffer_desc->setErrorOptions(MTL::CommandBufferErrorOptionEncoderExecutionStatus);
	auto* buffer = command_queue->commandBuffer(buffer_desc);

	auto drawCount = scene.meshlet_draws.size();
	if (drawCount > 0) {
		auto* visbuffer_pass_descriptor = MTL::RenderPassDescriptor::alloc()->init()->autorelease();
		visbuffer_pass_descriptor->setImageblockSampleLength(visbuffer_pass.gbuffer_tile_pipeline->imageblockSampleLength());

		auto* albedo_attachment = visbuffer_pass_descriptor->colorAttachments()->object(0);
		albedo_attachment->setLoadAction(MTL::LoadActionClear);
		albedo_attachment->setStoreAction(MTL::StoreActionStore);
		albedo_attachment->setTexture(visbuffer_pass.albedo_texture.get());
		albedo_attachment->setClearColor(MTL::ClearColor::Make(0., 0., 0., 1.));

		auto* normal_attachment = visbuffer_pass_descriptor->colorAttachments()->object(1);
		normal_attachment->setLoadAction(MTL::LoadActionClear);
		normal_attachment->setStoreAction(MTL::StoreActionStore);
		normal_attachment->setTexture(visbuffer_pass.normal_texture.get());
		normal_attachment->setClearColor(MTL::ClearColor::Make(0., 0., 0., 0.));

		auto* mr_attachment = visbuffer_pass_descriptor->colorAttachments()->object(2);
		mr_attachment->setLoadAction(MTL::LoadActionClear);
		mr_attachment->setStoreAction(MTL::StoreActionStore);
		mr_attachment->setTexture(visbuffer_pass.metallic_roughness_texture.get());
		mr_attachment->setClearColor(MTL::ClearColor::Make(0., 0., 0., 0.));

		auto* depthAttachment = visbuffer_pass_descriptor->depthAttachment();
		depthAttachment->setClearDepth(0.);
		depthAttachment->setLoadAction(MTL::LoadActionClear);
		depthAttachment->setStoreAction(MTL::StoreActionStore);
		depthAttachment->setTexture(visbuffer_pass.depth_texture.get());

		auto* encoder = buffer->renderCommandEncoder(visbuffer_pass_descriptor);
		encoder->setLabel(MTLSTR("Visbuffer raster"));

		encoder->setRenderPipelineState(visbuffer_pass.visbuffer_pipeline.get());
		encoder->setDepthStencilState(visbuffer_pass.depth_state.get());

		auto& sceneDrawBuffers = scene.draw_buffers[frameIndex];

		for (auto& meshes : scene.meshes) {
			encoder->useResource(meshes.mesh->vertex_index_buffer.get(), MTL::ResourceUsageRead);
			encoder->useResource(meshes.mesh->primitive_index_buffer.get(), MTL::ResourceUsageRead);
			encoder->useResource(meshes.mesh->position_buffer.get(), MTL::ResourceUsageRead);
			encoder->useResource(meshes.mesh->vertex_buffer.get(), MTL::ResourceUsageRead);
			encoder->useResource(meshes.mesh->meshlet_buffer.get(), MTL::ResourceUsageRead);
			for (auto& uv_buffer : meshes.mesh->uv_buffers)
				if (uv_buffer)
					encoder->useResource(uv_buffer.get(), MTL::ResourceUsageRead);
		}
		resource_table->encode_usage(encoder);

		encoder->setObjectBytes(&drawCount, sizeof drawCount, 0);
		encoder->setObjectBuffer(camera_buffers[frameIndex].get(), 0, 1);
		encoder->setObjectBuffer(sceneDrawBuffers.meshlet_draw_buffer.get(), 0, 2);
		encoder->setObjectBuffer(sceneDrawBuffers.transform_buffer.get(), 0, 3);
		encoder->setObjectBuffer(sceneDrawBuffers.primitive_buffer.get(), 0, 4);

		encoder->setMeshBuffer(sceneDrawBuffers.meshlet_draw_buffer.get(), 0, 0);
		encoder->setMeshBuffer(sceneDrawBuffers.transform_buffer.get(), 0, 1);
		encoder->setMeshBuffer(sceneDrawBuffers.primitive_buffer.get(), 0, 2);
		encoder->setMeshBuffer(camera_buffers[frameIndex].get(), 0, 3);
		encoder->setMeshBuffer(material_buffers[frameIndex].get(), 0, 4);

		encoder->setFragmentBuffer(material_buffers[frameIndex].get(), 0, 0);
		encoder->setFragmentBuffer(resource_table->sampled_image_buffer, 0, 1);

		// We cull manually per primitive in the mesh shader
		encoder->setCullMode(MTL::CullModeNone);

		encoder->drawMeshThreadgroups(
				MTL::Size((drawCount + shaders::maxMeshlets - 1) / shaders::maxMeshlets, 1, 1),
				MTL::Size(visbuffer_pass.visbuffer_pipeline->objectThreadExecutionWidth(), 1, 1),
				MTL::Size(visbuffer_pass.visbuffer_pipeline->meshThreadExecutionWidth(), 1, 1));

		encoder->setRenderPipelineState(visbuffer_pass.gbuffer_tile_pipeline.get());

		encoder->setTileBuffer(sceneDrawBuffers.meshlet_draw_buffer.get(), 0, 0);
		encoder->setTileBuffer(sceneDrawBuffers.transform_buffer.get(), 0, 1);
		encoder->setTileBuffer(sceneDrawBuffers.primitive_buffer.get(), 0, 2);
		encoder->setTileBuffer(camera_buffers[frameIndex].get(), 0, 3);
		encoder->setTileBuffer(material_buffers[frameIndex].get(), 0, 4);
		encoder->setTileBuffer(resource_table->sampled_image_buffer, 0, 5);

		encoder->dispatchThreadsPerTile(
			MTL::Size::Make(encoder->tileHeight(), encoder->tileWidth(), 1));

		encoder->endEncoding();
	}

	if (drawCount > 0) {
		auto* shading_pass_descriptor = MTL::RenderPassDescriptor::alloc()->init()->autorelease();
		shading_pass_descriptor->setImageblockSampleLength(visbuffer_pass.gbuffer_tile_pipeline->imageblockSampleLength());

		auto* color_attachment = shading_pass_descriptor->colorAttachments()->object(0);
		color_attachment->setLoadAction(MTL::LoadActionClear);
		color_attachment->setStoreAction(MTL::StoreActionStore);
		color_attachment->setTexture(drawable->texture());
		color_attachment->setClearColor(MTL::ClearColor::Make(0., 0., 0., 1.));

		auto* albedo_attachment = shading_pass_descriptor->colorAttachments()->object(1);
		albedo_attachment->setLoadAction(MTL::LoadActionLoad);
		albedo_attachment->setStoreAction(MTL::StoreActionDontCare);
		albedo_attachment->setTexture(visbuffer_pass.albedo_texture.get());
		albedo_attachment->setClearColor(MTL::ClearColor::Make(0., 0., 0., 1.));

		auto* normal_attachment = shading_pass_descriptor->colorAttachments()->object(2);
		normal_attachment->setLoadAction(MTL::LoadActionLoad);
		normal_attachment->setStoreAction(MTL::StoreActionDontCare);
		normal_attachment->setTexture(visbuffer_pass.normal_texture.get());
		normal_attachment->setClearColor(MTL::ClearColor::Make(0., 0., 0., 0.));

		auto* mr_attachment = shading_pass_descriptor->colorAttachments()->object(3);
		mr_attachment->setLoadAction(MTL::LoadActionLoad);
		mr_attachment->setStoreAction(MTL::StoreActionDontCare);
		mr_attachment->setTexture(visbuffer_pass.metallic_roughness_texture.get());
		mr_attachment->setClearColor(MTL::ClearColor::Make(0., 0., 0., 0.));

		auto* encoder = buffer->renderCommandEncoder(shading_pass_descriptor);
		encoder->setLabel(MTLSTR("Shading"));

		encoder->setRenderPipelineState(shading_pass.shading_pass_pipeline.get());

		encoder->setTileBuffer(camera_buffers[frameIndex].get(), 0, 0);
		encoder->setTileTexture(visbuffer_pass.depth_texture.get(), 0);

		encoder->dispatchThreadsPerTile(
			MTL::Size::Make(encoder->tileHeight(), encoder->tileWidth(), 1));

		encoder->endEncoding();
	}

	imgui_renderer->draw(buffer, drawable, get_render_resolution(), frameIndex, drawCount == 0);

	buffer->presentDrawable(drawable);

	// The completion handlers execute on another thread than our main thread, and the command buffer
	// might outlive this MtlRenderer class. Therefore, we need to keep a weak reference to this class
	// alive in the lambda capture instead of just blindly capturing `this`, which might not be a valid
	// object anymore when this executes.
	buffer->addCompletedHandler([weak = weak_from_this()](MTL::CommandBuffer* buffer) {
		ZoneScoped;
		if (auto renderer = std::dynamic_pointer_cast<meshlet_renderer>(weak.lock())) {
			dispatch_semaphore_signal(renderer->draw_semaphore);

			if (buffer->status() == MTL::CommandBufferStatusError) {
				auto* error = buffer->error();
				if (error) {
					fmt::print(stderr, "Command buffer error: {}\n", error->localizedDescription()->utf8String());

					auto* encoderInfos = static_cast<NS::Array*>(
							error->userInfo()->object(MTL::CommandBufferEncoderInfoErrorKey));
					for (std::size_t i = 0; i < encoderInfos->count(); ++i) {
						auto* info = static_cast<MTL::CommandBufferEncoderInfo*>(encoderInfos->object(i));
						for (std::size_t j = 0; j < info->debugSignposts()->count(); ++j) {
							auto* signpost = static_cast<NS::String*>(info->debugSignposts()->object(j));
							fmt::print(stderr, "Signpost {}: {}", j, signpost->utf8String());
						}
					}
				}

				renderer->command_buffer_exception = std::make_exception_ptr(
						std::runtime_error("Failed to execute command buffer"));
			}
		}
	});

	buffer->commit();

	pool->release();

	return true;
}
