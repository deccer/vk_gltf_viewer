#include <array>

#include <tracy/Tracy.hpp>

#include <fastgltf/util.hpp>
#include <vk_gltf_viewer/scheduler.hpp>

struct PinnedTaskRunLoop : enki::IPinnedTask {
	void Execute() override {
		while (!task_scheduler.GetIsShutdownRequested()) {
			// This thread will sleep until a new pinned task is available for the thread,
			// and then run it.
			task_scheduler.WaitForNewPinnedTasks();
			task_scheduler.RunPinnedTasks();
		}
	}
};

std::array<PinnedTaskRunLoop, fastgltf::to_underlying(PinnedThreadId::Count)> pinnedTaskRunners;

void initialize_scheduler() {
	ZoneScoped;
	auto pinnedThreadCount = fastgltf::to_underlying(PinnedThreadId::Count);

	enki::TaskSchedulerConfig config;
	config.numTaskThreadsToCreate += pinnedThreadCount;

	task_scheduler.Initialize(config);

	// Start the pinned tasks
	std::uint32_t normalThreads = task_scheduler.GetNumTaskThreads() - pinnedThreadCount;
	for (std::uint32_t i = 0; i < pinnedThreadCount; ++i) {
		pinnedTaskRunners[i].threadNum = i + normalThreads;
		task_scheduler.AddPinnedTask(&pinnedTaskRunners[i]);
	}
}

std::uint32_t getPinnedThreadNum(PinnedThreadId id) {
	assert(!task_scheduler.GetIsShutdownRequested());
	return (task_scheduler.GetNumTaskThreads() - fastgltf::to_underlying(PinnedThreadId::Count)) + fastgltf::to_underlying(id);
}
