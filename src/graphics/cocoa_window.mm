#import <QuartzCore/CAMetalLayer.h>

#define GLFW_EXPOSE_NATIVE_COCOA
#import <GLFW/glfw3.h>
#import <GLFW/glfw3native.h>

#include <tracy/Tracy.hpp>

#import <graphics/mtl_renderer.hpp>

CA::MetalLayer* graphics::metal::create_metal_layer(GLFWwindow* window) {
	ZoneScoped;
	auto layer = [CAMetalLayer layer];

	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	layer.drawableSize = CGSizeMake(width, height);
	layer.displaySyncEnabled = YES;
	layer.framebufferOnly = NO;

    layer.wantsExtendedDynamicRangeContent = YES;
    auto colorspace = CGColorSpaceCreateWithName(kCGColorSpaceDisplayP3);
    layer.colorspace = colorspace;
    CGColorSpaceRelease(colorspace);

	NSWindow* nswindow = glfwGetCocoaWindow(window);
	nswindow.contentView.layer = layer; // NOLINT
	nswindow.contentView.wantsLayer = TRUE;

	return (__bridge CA::MetalLayer*)layer;
}
