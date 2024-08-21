#include <utility>
#include <mutex>

#include <fmt/std.h>

#include <mesh_common.h>

#include <util.hpp>
#include <vk_gltf_viewer/assets.hpp>

#include <glm/gtc/type_ptr.hpp>

#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>

#include <meshoptimizer.h>

#define WUFFS_CONFIG__MODULE__AUX__IMAGE // For C++ image API
#include "wuffs-v0.4.c"

#define DDS_USE_STD_FILESYSTEM
#include <dds.hpp>

namespace fs = std::filesystem;
namespace fg = fastgltf;

#if defined(_MSC_VER) && !defined(__clang__)
std::size_t operator""UZ(unsigned long long int x) noexcept {
	return std::size_t(x);
}
#endif

struct BufferLoadTask : ExceptionTaskSet {
	fg::Asset& asset;
	fs::path folder;

	explicit BufferLoadTask(fg::Asset& asset, fs::path folder) : asset(asset), folder(std::move(folder)) {
		m_SetSize = fg::max(1UZ, asset.buffers.size());
	}
	void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};

void BufferLoadTask::ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) {
	ZoneScoped;
	if (asset.buffers.empty())
		return;

	for (auto i = range.start; i < range.end; ++i) {
		auto& buffer = asset.buffers[i];

		// The buffer data is already in CPU memory, we don't need to do anything.
		if (std::holds_alternative<fg::sources::Vector>(buffer.data) || std::holds_alternative<fg::sources::Array>(buffer.data) || std::holds_alternative<fg::sources::ByteView>(buffer.data)) {
			continue;
		}

		if (std::holds_alternative<fg::sources::Fallback>(buffer.data))
			continue; // Ignore these.

		// We only support loading from local URIs.
		assert(std::holds_alternative<fg::sources::URI>(buffer.data));

		const auto& uri = std::get<fg::sources::URI>(buffer.data);
		auto filePath = folder / uri.uri.fspath();
		std::ifstream file(filePath, std::ios::binary);
		if (!file) {
			throw std::runtime_error(fmt::format("Failed to open buffer: {}", filePath));
		}

		fg::StaticVector<std::byte> data(buffer.byteLength);
		file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size_bytes()));
		fg::sources::Array arraySource {
			std::move(data),
		};
		buffer.data = std::move(arraySource);
	}
}

/** Replacement buffer data adapter for fastgltf which supports decompressing with EXT_meshopt_compression */
struct CompressedBufferDataAdapter : enki::ITaskSet {
	fg::Asset& asset;

	std::vector<std::optional<fastgltf::StaticVector<std::byte>>> decompressedBuffers;

	enki::Dependency bufferLoadDependency;

	explicit CompressedBufferDataAdapter(fg::Asset& asset) : asset(asset) {
		m_SetSize = asset.bufferViews.size();
		m_MinRange = fastgltf::min(24U, m_SetSize);
		decompressedBuffers.resize(m_SetSize);
	}

	/** Get the data pointer of a loaded (possibly compressed) buffer */
	[[nodiscard]] static auto getData(const fastgltf::Buffer& buffer, std::size_t byteOffset, std::size_t byteLength) {
		ZoneScoped;
		using namespace fastgltf;
		return std::visit(visitor {
			[](auto&) -> span<const std::byte> {
				assert(false && "Tried accessing a buffer with no data, likely because no buffers were loaded. Perhaps you forgot to specify the LoadExternalBuffers option?");
				return {};
			},
			[](const sources::Fallback& fallback) -> span<const std::byte> {
				assert(false && "Tried accessing data of a fallback buffer.");
				return {};
			},
			[&](const sources::Array& array) -> span<const std::byte> {
				return span(array.bytes.data(), array.bytes.size_bytes());
			},
			[&](const sources::Vector& vec) -> span<const std::byte> {
				return span(vec.bytes.data(), vec.bytes.size());
			},
			[&](const sources::ByteView& bv) -> span<const std::byte> {
				return bv.bytes;
			},
		}, buffer.data).subspan(byteOffset, byteLength);
	}

	/** Decompress all buffer views and store them in this adapter */
	void ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) override {
		ZoneScoped;

		for (auto i = range.start; i < range.end; ++i) {
			auto& bufferView = asset.bufferViews[i];
			if (!bufferView.meshoptCompression) {
				continue;
			}

			// This is a compressed buffer view.
			// For the original implementation, see https://github.com/jkuhlmann/cgltf/pull/129#issue-739550034
			auto& mc = *bufferView.meshoptCompression;
			fastgltf::StaticVector<std::byte> result(mc.count * mc.byteStride);

			// Get the data span from the compressed buffer.
			auto data = getData(asset.buffers[mc.bufferIndex], mc.byteOffset, mc.byteLength);

			int rc = -1;
			switch (mc.mode) {
				using enum fg::MeshoptCompressionMode;
				case Attributes: {
					rc = meshopt_decodeVertexBuffer(result.data(), mc.count, mc.byteStride,
													reinterpret_cast<const unsigned char*>(data.data()), mc.byteLength);
					break;
				}
				case Triangles: {
					rc = meshopt_decodeIndexBuffer(result.data(), mc.count, mc.byteStride,
											  reinterpret_cast<const unsigned char*>(data.data()), mc.byteLength);
					break;
				}
				case Indices: {
					rc = meshopt_decodeIndexSequence(result.data(), mc.count, mc.byteStride,
												reinterpret_cast<const unsigned char*>(data.data()), mc.byteLength);
					break;
				}
			}

			//if (rc != 0)
			//	  return false;

			switch (mc.filter) {
				using enum fg::MeshoptCompressionFilter;
				case None:
					break;
				case Octahedral: {
					meshopt_decodeFilterOct(result.data(), mc.count, mc.byteStride);
					break;
				}
				case Quaternion: {
					meshopt_decodeFilterQuat(result.data(), mc.count, mc.byteStride);
					break;
				}
				case Exponential: {
					meshopt_decodeFilterExp(result.data(), mc.count, mc.byteStride);
					break;
				}
			}

			decompressedBuffers[i] = std::move(result);
		}
	}

	auto operator()([[maybe_unused]] const fastgltf::Asset& _, std::size_t bufferViewIdx) const {
		ZoneScoped;
		using namespace fastgltf;

		auto& bufferView = asset.bufferViews[bufferViewIdx];
		if (bufferView.meshoptCompression) {
			assert(decompressedBuffers.size() == asset.bufferViews.size());

			assert(decompressedBuffers[bufferViewIdx].has_value());
			return span(decompressedBuffers[bufferViewIdx]->data(), decompressedBuffers[bufferViewIdx]->size_bytes());
		}

		return getData(asset.buffers[bufferView.bufferIndex], bufferView.byteOffset, bufferView.byteLength);
	}
};

struct ImageLoadTask : ExceptionTaskSet {
	const fg::Asset& asset;
	std::shared_ptr<graphics::Renderer> renderer;

	std::vector<std::shared_ptr<graphics::Image>> images;

	explicit ImageLoadTask(const fg::Asset& asset, std::shared_ptr<graphics::Renderer> renderer) : asset(asset), renderer(std::move(renderer)) {
		images.resize(asset.images.size());
		m_SetSize = fg::max(1UZ, asset.images.size());
	}

	std::shared_ptr<graphics::Image> loadDefault(std::span<const std::byte> imageBytes);

	static fg::MimeType detectMimeType(std::span<const std::byte> data);
	std::shared_ptr<graphics::Image> load(std::span<const std::byte> imageBytes, fg::MimeType mimeType);
	void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};

std::shared_ptr<graphics::Image> ImageLoadTask::loadDefault(std::span<const std::byte> imageBytes) {
	ZoneScoped;

	wuffs_aux::DecodeImageCallbacks callbacks;
	wuffs_aux::sync_io::MemoryInput input(reinterpret_cast<const char*>(imageBytes.data()), imageBytes.size_bytes());
	auto decodeResult = wuffs_aux::DecodeImage(callbacks, input);
	if (!decodeResult.error_message.empty()) {
		throw std::runtime_error(decodeResult.error_message);
	}
	if (!decodeResult.pixbuf.pixcfg.pixel_format().is_interleaved()) {
		throw std::runtime_error("Cannot load non-interleaved PNG/JPEG images.");
	}

	std::uint32_t width = decodeResult.pixbuf.pixcfg.width(), height = decodeResult.pixbuf.pixcfg.height();

	auto imageData = std::span(
		reinterpret_cast<std::byte*>(decodeResult.pixbuf.plane(0).ptr),
		decodeResult.pixbuf.pixcfg.pixbuf_len());
	return renderer->createSharedImage(imageData, glm::u32vec2(width, height));
}

/** Sometimes, the mimeType field is unspecified. Here, we try and detect the mimeType if it's initially None */
fg::MimeType ImageLoadTask::detectMimeType(std::span<const std::byte> data) {
	ZoneScoped;

	// The glTF did not provide a mime type. Try to detect the image format using the header magic.
	// See https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#images for the table of patterns
	if (data[0] == std::byte{0xFF} && data[1] == std::byte{0xD8} && data[2] == std::byte{0xFF}) {
		return fg::MimeType::JPEG;
	}

	static constexpr auto pngMagic = std::to_array<std::uint8_t>({
		0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A
	});
	if (std::memcmp(data.data(), pngMagic.data(), pngMagic.size()) == 0) {
		return fg::MimeType::PNG;
	}

	if (*reinterpret_cast<const std::uint32_t*>(data.data()) == dds::DdsMagicNumber::DDS) {
		return fg::MimeType::DDS;
	}

	if (data[0] == std::byte{0xAB} && data[1] == std::byte{'K'} && data[2] == std::byte{'T'} && data[3] == std::byte{'X'}) {
		return fg::MimeType::KTX2;
	}

	throw std::runtime_error(
		fmt::format("Failed to detect image mime type while loading. Header magic: {0:x}", *reinterpret_cast<const std::uint32_t*>(data.data())));
}

std::shared_ptr<graphics::Image> ImageLoadTask::load(std::span<const std::byte> imageBytes, fg::MimeType mimeType) {
	ZoneScoped;
	if (mimeType == fg::MimeType::None) {
		mimeType = detectMimeType(imageBytes);
	}

	switch (mimeType) {
		using fg::MimeType;
		case MimeType::PNG:
		case MimeType::JPEG: {
			return loadDefault(imageBytes);
		}
		default: {
			throw std::runtime_error(fmt::format("Unsupported mime type for loading images: {}", fastgltf::to_underlying(mimeType)));
		}
	}
}

void ImageLoadTask::ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) {
	ZoneScoped;
	if (asset.images.empty())
		return;

	for (auto i = range.start; i < range.end; ++i) {
		const auto& image = asset.images[i];

		std::visit(fg::visitor {
			[]([[maybe_unused]] auto& arg) {
				throw std::runtime_error("Got an unexpected image data source. Can't load image.");
			},
			[&](const fg::sources::Array& array) {
				images[i] = load(std::span(array.bytes.data(), array.bytes.size()), array.mimeType);
			},
			[&](fg::sources::BufferView& bufferView) {
				auto& view = asset.bufferViews[bufferView.bufferViewIndex];
				auto& buffer = asset.buffers[view.bufferIndex];
				return std::visit(fg::visitor{
					[]([[maybe_unused]] auto& arg) {
						throw std::runtime_error("Got an unexpected image data source. Can't load image.");
					},
					[&](fg::sources::Array& array) {
						images[i] = load(std::span(array.bytes.data() + view.byteOffset, array.bytes.size()), bufferView.mimeType);
					}
				}, buffer.data);
			}
		}, image.data);
	}
}

/**
 * The TextureCreateTask is responsible for creating samplers and bundling them together into textures.
 * We use a set size of 1 here, since this is a process that is pretty much free and the thread scheduling
 * overhead would be too big in most cases.
 */
struct TextureCreateTask : ExceptionTaskSet {
	const fg::Asset& asset;
	std::shared_ptr<graphics::Renderer> renderer;

	std::vector<std::shared_ptr<graphics::Texture>> textures;

	enki::Dependency imageLoadDependency;

	explicit TextureCreateTask(const fg::Asset& asset, std::shared_ptr<graphics::Renderer> renderer) : asset(asset), renderer(std::move(renderer)) {
		textures.resize(asset.textures.size());
		m_SetSize = 1;
	}

	void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};

void TextureCreateTask::ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) {
	std::vector<std::shared_ptr<graphics::Sampler>> samplers(asset.samplers.size());

	for (std::size_t i = 0; auto& sampler : asset.samplers)
		samplers[i++] = renderer->createSharedSampler(sampler);

	auto* imageTask = dynamic_cast<const ImageLoadTask*>(imageLoadDependency.GetDependencyTask());
	for (std::size_t i = 0; auto& texture : asset.textures) {
		auto sampler = texture.samplerIndex.has_value() ? samplers[texture.samplerIndex.value()] : renderer->getDefaultSampler();
		textures[i++] = renderer->createSharedTexture(imageTask->images[*texture.imageIndex], sampler);
	}
}

struct MaterialLoadTask : enki::ITaskSet {
	const fg::Asset& asset;
	std::shared_ptr<graphics::Renderer> renderer;
	std::vector<graphics::MaterialIndex> materials;

	std::mutex materialMutex; // TODO: Should the renderer have this mutex instead?

	enki::Dependency textureDependency;

	explicit MaterialLoadTask(const fg::Asset& asset, std::shared_ptr<graphics::Renderer> renderer) noexcept : asset(asset), renderer(std::move(renderer)) {
		materials.resize(asset.materials.size());
		m_SetSize = fg::max(1UZ, asset.materials.size());
		m_MinRange = fg::min(16U, m_SetSize);
	}

	void ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};

void MaterialLoadTask::ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) {
	ZoneScoped;
	if (asset.materials.empty())
		return;

	auto* textureTask = dynamic_cast<const TextureCreateTask*>(textureDependency.GetDependencyTask());
	for (auto i = range.start; i < range.end; ++i) {
		auto& gltfMaterial = asset.materials[i];
		auto& pbr = gltfMaterial.pbrData;

		shaders::material_t material {
			.albedo_factor = glm::make_vec4(pbr.baseColorFactor.data()),
			.alpha_cutoff = gltfMaterial.alphaCutoff,
			.double_sided = gltfMaterial.doubleSided,
		};

		if (pbr.baseColorTexture) {
			material.albedo_index = textureTask->textures[pbr.baseColorTexture->textureIndex]->getHandle();
			material.albedo_uv_set = static_cast<std::uint32_t>(pbr.baseColorTexture->texCoordIndex);
			if (auto& transform = pbr.baseColorTexture->transform; transform) {
				material.uv_offset = glm::make_vec2(transform->uvOffset.data());
				material.uv_scale = glm::make_vec2(transform->uvScale.data());
				material.uv_rotation = transform->rotation;
			}
		}

		std::lock_guard lock(materialMutex);
		materials[i] = renderer->createMaterial(material);
	}
}

/**
 * Processes all glTF primitives into meshlets
 */
struct PrimitiveProcessingTask : ExceptionTaskSet {
	const fg::Asset& asset;
	const CompressedBufferDataAdapter& adapter;
	std::shared_ptr<graphics::Renderer> renderer;

	std::mutex meshMutex;
	std::vector<Mesh> meshes;
	std::vector<std::shared_ptr<graphics::Mesh>> primitives;

	enki::Dependency bufferDecompressDependency;
	enki::Dependency materialDependency;

	explicit PrimitiveProcessingTask(const fg::Asset& _asset, const CompressedBufferDataAdapter& _adapter, std::shared_ptr<graphics::Renderer> renderer) noexcept
			: asset(_asset), adapter(_adapter), renderer(std::move(renderer)) {
		ZoneScoped;
		m_SetSize = fg::max(1UZ, asset.meshes.size());
		meshes.resize(asset.meshes.size());
		primitives.reserve(meshes.size()); // Is definitely not enough, but we'll just live with it.
	}

	void processPrimitive(std::uint64_t primitiveIdx, const fg::Primitive& primitive);
	void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};

glm::vec3 getAccessorMinMax(const decltype(fg::Accessor::min)& values) {
	return std::visit(fg::visitor {
		[](const auto& arg) {
			return glm::vec3();
		},
		[&](const FASTGLTF_STD_PMR_NS::vector<double>& values) {
			assert(values.size() == 3);
			return glm::fvec3(values[0], values[1], values[2]);
		},
		[&](const FASTGLTF_STD_PMR_NS::vector<std::int64_t>& values) {
			assert(values.size() == 3);
			return glm::fvec3(values[0], values[1], values[2]);
		},
	}, values);
}

void PrimitiveProcessingTask::processPrimitive(std::uint64_t primitiveIdx, const fg::Primitive& gltfPrimitive) {
	ZoneScoped;
	// The code says this is possible, the spec says otherwise.
	auto* positionIt = gltfPrimitive.findAttribute("POSITION");
	assert(positionIt != gltfPrimitive.attributes.cend());

	auto& posAccessor = asset.accessors[positionIt->accessorIndex];

	glm::fvec3 aabbCenter, aabbExtents;
	{
		auto primitiveMin = getAccessorMinMax(posAccessor.min);
		auto primitiveMax = getAccessorMinMax(posAccessor.max);
		aabbCenter = (primitiveMin + primitiveMax) / 2.f;
		aabbExtents = primitiveMax - aabbCenter;
	}

	std::vector positions(std::from_range, fastgltf::iterateAccessor<glm::fvec3>(asset, posAccessor));

	assert(gltfPrimitive.indicesAccessor.has_value());
	auto& idxAccessor = asset.accessors[gltfPrimitive.indicesAccessor.value()];

	std::vector<graphics::index_t> indices(idxAccessor.count);
	fastgltf::copyFromAccessor<graphics::index_t>(
			asset, idxAccessor, indices.data(), adapter);

	std::vector vertices(posAccessor.count, shaders::vertex_t {
		.color = glm::u8vec4(255),
	});

	if (auto* normalAttribute = gltfPrimitive.findAttribute("NORMAL"); normalAttribute != gltfPrimitive.attributes.end()) {
		fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, asset.accessors[normalAttribute->accessorIndex], [&](glm::vec3 val, std::size_t idx) {
			vertices[idx].normal = val;
		}, adapter);
	} else {
		// Generate basic smooth vertex normals. As we quantize the normals, we have to store them first to normalize them afterwards.
		fastgltf::StaticVector<glm::vec3> normals(posAccessor.count, glm::vec3(0.f));
		for (std::uint32_t idx = 0; idx < idxAccessor.count / 3; ++idx) {
			auto i = indices[idx * 3];
			auto i1 = indices[idx * 3 + 1];
			auto i2 = indices[idx * 3 + 2];

			auto v1 = positions[i1] - positions[i];
			auto v2 = positions[i2] - positions[i];
			auto val = glm::normalize(glm::cross(v1, v2));

			normals[i] += val;
			normals[i1] += val;
			normals[i2] += val;
		}
		for (std::size_t i = 0; i < normals.size(); ++i) {
			vertices[i].normal = normals[i];
		}
	}

	if (auto* colorAttribute = gltfPrimitive.findAttribute("COLOR_0"); colorAttribute != gltfPrimitive.attributes.end()) {
		// The glTF spec allows VEC3 and VEC4 for COLOR_n, with VEC3 data having to be extended with 1.0f for the fourth component.
		auto& colorAccessor = asset.accessors[colorAttribute->accessorIndex];
		if (colorAccessor.type == fastgltf::AccessorType::Vec4) {
			fastgltf::iterateAccessorWithIndex<glm::vec4>(asset, colorAccessor, [&](glm::vec4 val, std::size_t idx) {
				val = glm::clamp(val, 0.0f, 1.0f);
				vertices[idx].color = glm::u8vec4(val * 255.f);
			}, adapter);
		} else if (colorAccessor.type == fastgltf::AccessorType::Vec3) {
			fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, colorAccessor, [&](glm::vec3 val, std::size_t idx) {
				val = glm::clamp(val, 0.0f, 1.0f);
				vertices[idx].color = glm::u8vec4(glm::vec4(val, 1.0f) * 255.f);
			}, adapter);
		}
	}

	std::array<std::vector<glm::fvec2>, shaders::max_uv_sets> uvs;

	// Initialize the first UV set to zeroes as a fallback
	uvs[0] = std::vector(posAccessor.count, glm::fvec2(0.f));

	for (const auto& [attribute, accessor_index] : gltfPrimitive.attributes) {
		// TODO: Also use this loop to load the other attributes, saving some iteration time?
		if (!attribute.starts_with("TEXCOORD_")) {
			continue;
		}

		// The spec requires indices to be just single positive decimal integers.
		static_assert(shaders::max_uv_sets <= 9, "The following code assumes single digit UV sets only.");
		auto idx = attribute[9] - '0';
		assert(idx < uvs.size() && idx >= 0);

		auto& accessor = asset.accessors[accessor_index];
		uvs[idx] = std::vector(std::from_range, fastgltf::iterateAccessor<glm::fvec2>(asset, accessor));
	}

	auto* materialTask = dynamic_cast<const MaterialLoadTask*>(materialDependency.GetDependencyTask());
	auto materialIndex = gltfPrimitive.materialIndex.has_value()
		? materialTask->materials[*gltfPrimitive.materialIndex]
		: renderer->getDefaultMaterialIndex();

	auto mesh = renderer->createSharedMesh(
			positions, vertices,
			transform_array(uvs, [](auto& buffer) { return std::span<const glm::fvec2>(buffer); }),
			indices,
			aabbCenter, aabbExtents,
			materialIndex);

	{
		std::lock_guard lock(meshMutex);
		primitives[primitiveIdx] = std::move(mesh);
	}
}

void PrimitiveProcessingTask::ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) {
	ZoneScoped;
	if (asset.meshes.empty())
		return;

	for (auto i = range.start; i < range.end; ++i) {
		auto& gltfMesh = asset.meshes[i];

		// Create the mesh/primitive CPU structures. We don't care about the order of the primitive buffer structures,
		// and therefore just append the primitives to the end of the vector. The mesh vector still needs to match the
		// order in the glTF asset though.
		{
			std::lock_guard lock(meshMutex);
			auto& mesh = meshes[i];
			mesh.primitiveIndices.resize(gltfMesh.primitives.size());

			for (std::size_t j = 0; j < gltfMesh.primitives.size(); ++j) {
				mesh.primitiveIndices[j] = primitives.size();
				primitives.emplace_back();
			}
		}

		for (std::size_t j = 0; auto& gltfPrimitive : gltfMesh.primitives) {
			processPrimitive(meshes[i].primitiveIndices[j++], gltfPrimitive);
		}
	}
}

AssetLoadTask::AssetLoadTask(std::shared_ptr<graphics::Renderer> renderer, fs::path path) : renderer(std::move(renderer)), assetPath(std::move(path)) {
	m_SetSize = 1;
}

std::shared_ptr<fastgltf::Asset> AssetLoadTask::loadGltf() {
	ZoneScoped;
	auto file = fg::MappedGltfFile::FromPath(assetPath);
	if (!bool(file)) {
		throw std::runtime_error("Failed to open glTF file");
	}

	static constexpr auto gltfOptions = fg::Options::GenerateMeshIndices
		| fg::Options::DecomposeNodeMatrices
		| fg::Options::LoadExternalImages;

	static constexpr auto gltfExtensions = fg::Extensions::EXT_meshopt_compression
		| fg::Extensions::KHR_mesh_quantization
		| fg::Extensions::KHR_texture_transform
		| fg::Extensions::KHR_texture_basisu
		| fg::Extensions::MSFT_texture_dds
		| fg::Extensions::KHR_lights_punctual;

	fg::Parser parser(gltfExtensions);
	parser.setUserPointer(this);

	// Load the glTF
	auto loadedAsset = parser.loadGltf(file.get(), assetPath.parent_path(), gltfOptions);
	if (!loadedAsset) {
		throw std::runtime_error(fmt::format("Failed to load glTF: {}", fg::getErrorMessage(loadedAsset.error())));
	}
	return std::make_shared<fg::Asset>(std::move(loadedAsset.get()));
}

void AssetLoadTask::ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) {
	ZoneScoped;
	asset = loadGltf();

	BufferLoadTask bufferLoadTask(*asset, assetPath.parent_path());

	CompressedBufferDataAdapter bufferDecompressTask(*asset);
	bufferDecompressTask.SetDependency(bufferDecompressTask.bufferLoadDependency, &bufferLoadTask);

	ImageLoadTask imageLoadTask(*asset, renderer);

	TextureCreateTask textureCreateTask(*asset, renderer);
	textureCreateTask.SetDependency(textureCreateTask.imageLoadDependency, &imageLoadTask);

	MaterialLoadTask materialLoadTask(*asset, renderer);
	materialLoadTask.SetDependency(materialLoadTask.textureDependency, &textureCreateTask);

	PrimitiveProcessingTask primitiveTask(*asset, bufferDecompressTask, renderer);
	primitiveTask.SetDependency(primitiveTask.bufferDecompressDependency, &bufferDecompressTask);
	primitiveTask.SetDependency(primitiveTask.materialDependency, &materialLoadTask);

	taskScheduler.AddTaskSetToPipe(&bufferLoadTask);
	taskScheduler.AddTaskSetToPipe(&imageLoadTask);

	taskScheduler.WaitforTask(&bufferDecompressTask);
	animations.resize(asset->animations.size());
	for (std::size_t i = 0; i < asset->animations.size(); ++i) {
		auto& gltfAnimation = asset->animations[i];
		auto& animation = animations[i];

		animation.channels.reserve(gltfAnimation.channels.size());
		for (auto& channel : gltfAnimation.channels) {
			animation.channels.emplace_back(channel);
		}

		animation.samplers.reserve(gltfAnimation.samplers.size());
		for (auto& gltfSampler : gltfAnimation.samplers) {
			auto& sampler = animation.samplers.emplace_back(*asset, gltfSampler);

			auto& inputAccessor = asset->accessors[gltfSampler.inputAccessor];
			sampler.input.resize(inputAccessor.count);
			fastgltf::copyFromAccessor<float>(*asset, inputAccessor, sampler.input.data(), bufferDecompressTask);

			auto& outputAccessor = asset->accessors[gltfSampler.outputAccessor];
			sampler.values.resize(outputAccessor.count * sampler.componentCount);
			fastgltf::copyComponentsFromAccessor<float>(*asset, outputAccessor, sampler.values.data(), bufferDecompressTask);
		}
	}

	taskScheduler.WaitforTask(&primitiveTask);
	taskScheduler.WaitforTask(&imageLoadTask);
	taskScheduler.WaitforTask(&materialLoadTask);

	meshes = std::move(primitiveTask.meshes);
	primitives = std::move(primitiveTask.primitives);
	textures = std::move(textureCreateTask.textures);
}
