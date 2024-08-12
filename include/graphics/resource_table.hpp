#pragma once

#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <vulkan/vk.hpp>

#if defined(VKV_METAL)
#include <Foundation/NSSharedPtr.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLBuffer.hpp>
#include <Metal/MTLCommandEncoder.hpp>
#endif

#include <resource_table.h>

struct Device;

namespace std {
	template <typename T>
	struct hash<NS::SharedPtr<T>> {
		std::size_t operator()(const NS::SharedPtr<T>& t) const noexcept {
			return std::hash<T*>()(t.get());
		}
	};
}

namespace graphics {
	class ResourceTable {
	protected:
		/** Each bit of each integer represents a boolean whether that array element is free or not */
		std::vector<std::uint64_t> sampledImageBitmap;
		std::vector<std::uint64_t> storageImageBitmap;
		std::mutex bitmapMutex;

		shaders::ResourceTableHandle findFirstFreeHandle(std::vector<std::uint64_t> &bitmap);

		void freeHandle(std::vector<std::uint64_t> &bitmap, shaders::ResourceTableHandle handle);

	public:
		explicit ResourceTable() = default;
		virtual ~ResourceTable() noexcept = default;

		virtual void removeSampledImageHandle(shaders::ResourceTableHandle handle) noexcept;
	};

	namespace vulkan {
		class VkResourceTable : public ResourceTable {
			std::reference_wrapper<Device> device;

			std::vector<std::uint64_t> storageImageBitmap;

			VkDescriptorPool pool = VK_NULL_HANDLE;
			VkDescriptorSetLayout layout = VK_NULL_HANDLE;
			VkDescriptorSet set = VK_NULL_HANDLE;

		public:
			explicit VkResourceTable(Device& device);
			~VkResourceTable() noexcept override;

			[[nodiscard]] auto getLayout() const noexcept -> const VkDescriptorSetLayout& {
				return layout;
			}

			[[nodiscard]] auto getSet() const noexcept -> const VkDescriptorSet& {
				return set;
			}

			[[nodiscard]] shaders::ResourceTableHandle allocateStorageImage(VkImageView view, VkImageLayout imageLayout) noexcept;
			[[nodiscard]] shaders::ResourceTableHandle allocateSampledImage(VkImageView view, VkImageLayout imageLayout, VkSampler sampler) noexcept;

			void removeStorageImageHandle(shaders::ResourceTableHandle handle) noexcept;
		};
	}

#if defined(VKV_METAL)
	namespace metal {
		class MtlResourceTable : public ResourceTable {
			struct SampledTextureEntry {
				MTL::ResourceID tex;
				MTL::ResourceID sampler;
			};

			NS::SharedPtr<MTL::Device> device;

			/** List of resources to make resident */
			std::unordered_map<shaders::ResourceTableHandle, NS::SharedPtr<MTL::Resource>> resources;

		public:
			MTL::Buffer* sampledImageBuffer = nullptr;

		public:
			explicit MtlResourceTable(NS::SharedPtr<MTL::Device> device);
			~MtlResourceTable() noexcept override;

			[[nodiscard]] shaders::ResourceTableHandle allocateSampledImage(NS::SharedPtr<MTL::Texture> texture, NS::SharedPtr<MTL::SamplerState> sampler) noexcept;

			void removeSampledImageHandle(shaders::ResourceTableHandle handle) noexcept override;

			template <typename T>
			void encodeUsage(T* encoder) {
				ZoneScoped;
				for (auto& [_, resource] : resources) {
					assert(resource.get() != nullptr);
					encoder->useResource(resource.get(), MTL::ResourceUsageRead);
				}
			}
		};
	}
#endif
} // namespace graphics
