#include <tuple>
#include <chrono>
#include <limits>
#include <vector>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <volk/volk.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
namespace lut = labutils;

#include "baked_model.hpp"
#include "load_data_to_vk.h"
#include <iostream>


namespace
{
	using Clock_ = std::chrono::steady_clock;
	using Secondsf_ = std::chrono::duration<float, std::ratio<1>>;

	namespace cfg
	{
		// Compiled shader code for the graphics pipeline
		// See sources in exercise4/shaders/*. 
#		define SHADERDIR_ "assets/cw3/shaders/"
		constexpr char const* kVertShaderPath = SHADERDIR_ "default.vert.spv";
		constexpr char const* kFragShaderPath = SHADERDIR_ "default.frag.spv";
		constexpr char const* kFullScreenVertShaderPath = SHADERDIR_ "fullscreen.vert.spv";
		constexpr char const* kFullScreenFragShaderPath = SHADERDIR_ "fullscreen.frag.spv";
		constexpr char const* kBloomH_FragShaderPath = SHADERDIR_ "hbloom.frag.spv";
		constexpr char const* kBloomV_FragShaderPath = SHADERDIR_ "vbloom.frag.spv";
#		undef SHADERDIR_

#		define ASSETDIR_ "assets/cw3/"
		constexpr char const* kBakedModelPath = ASSETDIR_"ship.comp5822mesh";
#		undef ASSETDIR_

		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;


		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear = 0.01f;
		constexpr float kCameraFar = 100.f;

		constexpr auto kCameraFov = 60.0_degf;

		// Camera settings.
		// These are determined empirically (i.e., by testing and picking something
		// that felt OK).
		//
		// General rule: for debugging, you want to be able to move around quickly
		// in the scene (but slow down if necessary). The exact settings here
		// depend on the scene scale and similar settings.
		constexpr float kCameraBaseSpeed = 1.7f; // units/second
		constexpr float kCameraFastMult = 5.f; // speed multiplier
		constexpr float kCameraSlowMult = 0.05f; // speed multiplier

		constexpr float kCameraMouseSensitivity = 0.01f; // radians per pixel

		int frameCounter = 0;
	}

	// GLFW callbacks
	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);
	void glfw_callback_button(GLFWwindow*, int, int, int);
	void glfw_callback_motion(GLFWwindow*, double, double);

	enum class EInputState
	{
		forward,
		backward,
		strafeLeft,
		strafeRight,
		levitate,
		sink,
		fast,
		slow,
		mousing,
		lightRotate,
		max
	};

	struct  UserState
	{
		bool inputMap[std::size_t(EInputState::max)] = {};

		float mouseX = 0.f, mouseY = 0.f;
		float previousX = 0.f, previousY = 0.f;

		bool wasMousing = false;
		//bool wasLightOrbiting = false;

		glm::mat4 camera2world = glm::identity<glm::mat4>();
		glm::vec4 light_pos0 = glm::vec4(1, 6, -4, 1);
		glm::vec4 light_pos1 = glm::vec4(-2, 5, 0, 1);
		glm::vec4 light_pos2 = glm::vec4(-4, 3, -2, 1);
	};

	//Also declare a update_user_state function to update the state based on the elapsed time:
	void update_user_state(UserState&, float aElapsedTime);


	// Uniform data
	namespace glsl
	{
		struct alignas(16) SceneUniform
		{
			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projCam;
			glm::vec3 cameraPos;
			float _pad0;
			glm::vec4 lightPos[3];
			glm::vec4 lightColor[3];
		};


		// We want to use vkCmdUpdateBuffer() to update the contents of our uniform
		// buffers. vkCmdUpdateBuffer() has a number of requirements, including
		// the two below. See https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCmdUpdateBuffer.html
		static_assert(sizeof(SceneUniform) <= 65536, "SceneUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(SceneUniform) % 4 == 0, "SceneUniform size must be a multiple of 4 bytes");
	}

	// Helpers:
	lut::RenderPass create_render_pass_A(lut::VulkanWindow const&);
	lut::RenderPass create_render_pass_B(lut::VulkanWindow const&);
	lut::RenderPass create_render_pass_Bloom(lut::VulkanWindow const&);
	//renderPass bloom

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_material_descriptor_layout(lut::VulkanWindow const& aWindow);
	lut::DescriptorSetLayout create_fullScreen_descriptor_layout(lut::VulkanWindow const& aWindow);
	lut::DescriptorSetLayout create_bloom_descriptor_layout(lut::VulkanWindow const& aWindow);

	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout, VkDescriptorSetLayout);
	lut::PipelineLayout create_post_processing_pipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout);
	lut::PipelineLayout create_final_pipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout, VkDescriptorSetLayout);

	lut::Pipeline create_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_post_processing_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_bloom_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout, const char*);

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const&, lut::Allocator const&);

	void create_swapchain_framebuffers(
		lut::VulkanWindow const&,
		VkRenderPass,
		std::vector<lut::Framebuffer>&
	);
	void create_Intermediate_framebuffers(lut::VulkanWindow const& aWindow, 
VkRenderPass aRenderPass, lut::Framebuffer& aFramebuffers, VkImageView aColorView, VkImageView aBrightColorView, VkImageView aDepthView);

	void create_bloom_framebuffer(lut::VulkanWindow const& aWindow,VkRenderPass aRenderPass, lut::Framebuffer& aFramebuffers,  VkImageView aView);



	void update_scene_uniforms(
		glsl::SceneUniform&,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight,
		UserState aState
	);
	void record_commandsA(VkCommandBuffer aCmdBuff, VkRenderPass aRenderPassA,
		VkPipeline aGraphicsPipe, VkExtent2D const& aImageExtent, VkBuffer aSceneUBO, glsl::SceneUniform const& aSceneUniform,
		VkPipelineLayout aGraphicsLayout, VkDescriptorSet aSceneDescriptors, ModelPack& aModel, VkFramebuffer aIntmdtFramebuffer);

	void record_commandsB(VkCommandBuffer aCmdBuff, VkFramebuffer aFramebuffer, VkExtent2D const& aImageExtent,
		VkRenderPass aRenderPassB, VkPipeline aPostPipe, VkPipelineLayout aPostPipeLayout, VkDescriptorSet aFullscreenDesc,
		VkFramebuffer aBloomHFramebuffer, VkFramebuffer aBloomVFramebuffer, VkRenderPass aBloomHPass, VkRenderPass aBloomVPass, VkPipelineLayout aBloomPipeLayout,  
		VkPipeline aBloomHPipe, VkPipeline aBloomVPipe, VkDescriptorSet aBrightColorDesc, VkDescriptorSet aBloomHDesc, VkDescriptorSet aBloomVDesc, VkQueryPool aQueryPool);

	void submit_commandsA(lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence);

	void submit_commandsB(
		lut::VulkanWindow const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);
	void present_results(
		VkQueue,
		VkSwapchainKHR,
		std::uint32_t aImageIndex,
		VkSemaphore,
		bool& aNeedToRecreateSwapchain
	);

	
}


int main() try
{
	// Create Vulkan Window
	auto window = lut::make_vulkan_window();

	// Configure the GLFW window
	UserState state{};

	glfwSetWindowUserPointer(window.window, &state);

	glfwSetKeyCallback(window.window, &glfw_callback_key_press);
	glfwSetMouseButtonCallback(window.window, &glfw_callback_button);
	glfwSetCursorPosCallback(window.window, &glfw_callback_motion);


	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator(window);

	// Intialize resources
	lut::RenderPass renderPass = create_render_pass_A(window);
	lut::RenderPass renderPassB = create_render_pass_B(window);
	lut::RenderPass bloomHorizontalPass = create_render_pass_Bloom(window);
	lut::RenderPass bloomVerticalPass = create_render_pass_Bloom(window); 

	//TODO- (Section 3) create scene descriptor set layout
	lut::DescriptorSetLayout sceneLayout = create_scene_descriptor_layout(window);
	lut::DescriptorSetLayout objectLayout = create_material_descriptor_layout(window);
	lut::DescriptorSetLayout postDescLayout = create_fullScreen_descriptor_layout(window);
	lut::DescriptorSetLayout bloomDescLayout = create_bloom_descriptor_layout(window);

	lut::PipelineLayout pipeLayout = create_pipeline_layout(window, sceneLayout.handle, objectLayout.handle);
	lut::Pipeline pipe = create_pipeline(window, renderPass.handle, pipeLayout.handle);

	lut::PipelineLayout bloomPipeLayout = create_post_processing_pipeline_layout(window, bloomDescLayout.handle); 
	lut::Pipeline bloomHorizontalPipe = create_bloom_pipeline(window, bloomHorizontalPass.handle, bloomPipeLayout.handle, cfg::kBloomH_FragShaderPath); 
	lut::Pipeline bloomVerticalPipe = create_bloom_pipeline(window, bloomVerticalPass.handle, bloomPipeLayout.handle, cfg::kBloomV_FragShaderPath); 

	//lut::PipelineLayout postPipeLayout = create_post_processing_pipeline_layout(window, postDescLayout.handle);
	lut::PipelineLayout postPipeLayout = create_final_pipeline_layout(window, bloomDescLayout.handle, bloomDescLayout.handle);
	lut::Pipeline postPipe = create_post_processing_pipeline(window, renderPassB.handle, postPipeLayout.handle);


	auto [depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);

	//create intermediate textures
	lut::Image intermediateImage = lut::create_intermediate_image_texture2d(allocator, window.swapchainExtent.width, window.swapchainExtent.height,
		VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
	lut::ImageView intermediateView = lut::create_image_view_texture2d(window, intermediateImage.image, VK_FORMAT_R16G16B16A16_SFLOAT);

	//create intermediate bright textures
	lut::Image intermediateBrightColorImage = lut::create_intermediate_image_texture2d(allocator, window.swapchainExtent.width, window.swapchainExtent.height, 
		VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT); 
	lut::ImageView intermediateBrightColorView = lut::create_image_view_texture2d(window, intermediateBrightColorImage.image, VK_FORMAT_R16G16B16A16_SFLOAT);

	//create intermediate horizontal bloom textures
	lut::Image bloomHorizontalImage = lut::create_intermediate_image_texture2d(allocator, window.swapchainExtent.width, window.swapchainExtent.height, 
		VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT); 
	lut::ImageView bloomHorizontalView = lut::create_image_view_texture2d(window, bloomHorizontalImage.image, VK_FORMAT_R16G16B16A16_SFLOAT);

	//create intermediate vertical bloom textures
	lut::Image bloomVerticalImage = lut::create_intermediate_image_texture2d(allocator, window.swapchainExtent.width, window.swapchainExtent.height, 
		VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT); 
	lut::ImageView bloomVerticalView = lut::create_image_view_texture2d(window, bloomVerticalImage.image, VK_FORMAT_R16G16B16A16_SFLOAT);
	 

	lut::Framebuffer intermediateFramebuffer;
	create_Intermediate_framebuffers(window, renderPass.handle, intermediateFramebuffer, intermediateView.handle, intermediateBrightColorView.handle, depthBufferView.handle);

	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers(window, renderPassB.handle, framebuffers);

	lut::Framebuffer bloomHorizontalFramebuffer, bloomVerticalFramebuffer;
	create_bloom_framebuffer(window, bloomHorizontalPass.handle, bloomHorizontalFramebuffer, bloomHorizontalView.handle);
	create_bloom_framebuffer(window, bloomVerticalPass.handle, bloomVerticalFramebuffer, bloomVerticalView.handle); 


	lut::CommandPool cpool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	VkCommandBuffer intermediateCmdBuffer = lut::alloc_command_buffer(window,cpool.handle);
	lut::Fence intermediateCbfence = lut::create_fence(window, 0);


	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;

	for (std::size_t i = 0; i < framebuffers.size(); ++i)
	{
		cbuffers.emplace_back(lut::alloc_command_buffer(window, cpool.handle));
		cbfences.emplace_back(lut::create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT));
	}


	lut::Semaphore imageAvailable = lut::create_semaphore(window);
	lut::Semaphore renderFinished = lut::create_semaphore(window);


	ModelPack ourModel;
	BakedModel bakedModel;
	lut::DescriptorPool dPool = lut::create_descriptor_pool(window);
	lut::Sampler defaultSampler = lut::create_default_sampler(window);
	{
		lut::CommandPool loadCmdPool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
		bakedModel = load_baked_model(cfg::kBakedModelPath);
		ourModel = set_up_model(window, allocator, bakedModel, loadCmdPool.handle, dPool.handle, defaultSampler.handle, objectLayout.handle);
	}

	// create scene uniform buffer with lut::create_buffer()
	lut::Buffer sceneUBO = lut::create_buffer(allocator,
		sizeof(glsl::SceneUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);


	// allocate descriptor set for uniform buffer
	VkDescriptorSet sceneDescriptors = lut::alloc_desc_set(window, dPool.handle, sceneLayout.handle);


	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo sceneUboInfo{};
		sceneUboInfo.buffer = sceneUBO.buffer;
		sceneUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = sceneDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &sceneUboInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	VkDescriptorSet quadTexDescriptor = lut::alloc_desc_set(window, dPool.handle, bloomDescLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};
		VkDescriptorImageInfo imageInfo[1]{};
		imageInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfo[0].imageView = intermediateView.handle;
		imageInfo[0].sampler = defaultSampler.handle;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = quadTexDescriptor;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &imageInfo[0];


		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	VkDescriptorSet brightColorDesc = lut::alloc_desc_set(window, dPool.handle, bloomDescLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};  
		VkDescriptorImageInfo imageInfo[1]{};  
		imageInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;  
		imageInfo[0].imageView = intermediateBrightColorView.handle;   
		imageInfo[0].sampler = defaultSampler.handle; 

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; 
		desc[0].dstSet = brightColorDesc;
		desc[0].dstBinding = 0; 
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; 
		desc[0].descriptorCount = 1; 
		desc[0].pImageInfo = &imageInfo[0];
		 
		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	VkDescriptorSet bloomHorizontalDesc = lut::alloc_desc_set(window, dPool.handle, bloomDescLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{}; 
		VkDescriptorImageInfo imageInfo[1]{}; 
		imageInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; 
		imageInfo[0].imageView = bloomHorizontalView.handle;  
		imageInfo[0].sampler = defaultSampler.handle; 

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = bloomHorizontalDesc;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &imageInfo[0];

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}


	VkDescriptorSet bloomVerticalDesc = lut::alloc_desc_set(window, dPool.handle, bloomDescLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};
		VkDescriptorImageInfo imageInfo[1]{};
		imageInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfo[0].imageView = bloomVerticalView.handle;
		imageInfo[0].sampler = defaultSampler.handle;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = bloomVerticalDesc;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &imageInfo[0];

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}


	// create query pool
	VkQueryPoolCreateInfo queryPoolCreateInfo{}; 
	queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO; 
	queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP; 
	queryPoolCreateInfo.queryCount = 2; 

	VkQueryPool queryPool; 
	if (auto const res = vkCreateQueryPool(window.device, &queryPoolCreateInfo, nullptr, &queryPool); VK_SUCCESS != res) { 
		lut::Error("Unable to create query pool\n""vkCreateQueryPool() returned %s", lut::to_string(res).c_str()); 
	}


	// Application main loop
	bool recreateSwapchain = false;

	//timing
	auto previousClock = Clock_::now();

	while (!glfwWindowShouldClose(window.window))
	{
		glfwPollEvents(); // or: glfwWaitEvents()

		// Recreate swap chain?
		if (recreateSwapchain)
		{
			//TODO: (Section 1) re-create swapchain and associated resources - see Exercise 3!

			// We need to destroy several objects, which may still be in use by
			// the GPU. Therefore, first wait for the GPU to finish processing
			vkDeviceWaitIdle(window.device);

			//Recreate them
			auto const changes = recreate_swapchain(window);

			if (changes.changedFormat)
			{
				renderPass = create_render_pass_A(window);
				bloomHorizontalPass = create_render_pass_Bloom(window);
				bloomVerticalPass = create_render_pass_Bloom(window);
				renderPassB = create_render_pass_B(window);
			}


			if (changes.changedSize)
			{
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);
				intermediateImage = lut::create_intermediate_image_texture2d(allocator, window.swapchainExtent.width, window.swapchainExtent.height, 
					VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT); 
				intermediateView = lut::create_image_view_texture2d(window, intermediateImage.image, VK_FORMAT_R16G16B16A16_SFLOAT);

				//create intermediate bright textures
				intermediateBrightColorImage = lut::create_intermediate_image_texture2d(allocator, window.swapchainExtent.width, window.swapchainExtent.height,
					VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
				intermediateBrightColorView = lut::create_image_view_texture2d(window, intermediateBrightColorImage.image, VK_FORMAT_R16G16B16A16_SFLOAT);

				//create intermediate horizontal bloom textures
				bloomHorizontalImage = lut::create_intermediate_image_texture2d(allocator, window.swapchainExtent.width, window.swapchainExtent.height, 
					VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT); 
				bloomHorizontalView = lut::create_image_view_texture2d(window, bloomHorizontalImage.image, VK_FORMAT_R16G16B16A16_SFLOAT); 

				//create intermediate vertical bloom textures
				bloomVerticalImage = lut::create_intermediate_image_texture2d(allocator, window.swapchainExtent.width, window.swapchainExtent.height, 
					VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT); 
				bloomVerticalView = lut::create_image_view_texture2d(window, bloomVerticalImage.image, VK_FORMAT_R16G16B16A16_SFLOAT); 

			}

			create_Intermediate_framebuffers(window, renderPass.handle, intermediateFramebuffer, intermediateView.handle, intermediateBrightColorView.handle, depthBufferView.handle);
			create_bloom_framebuffer(window, bloomHorizontalPass.handle, bloomHorizontalFramebuffer, bloomHorizontalView.handle); 
			create_bloom_framebuffer(window, bloomVerticalPass.handle, bloomVerticalFramebuffer, bloomVerticalView.handle); 

			framebuffers.clear();
			create_swapchain_framebuffers(window, renderPassB.handle, framebuffers);

			//update the intermediate descriptor
			{
				VkWriteDescriptorSet desc[1]{};
				VkDescriptorImageInfo imageInfo[1]{};
				imageInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				imageInfo[0].imageView = intermediateView.handle;
				imageInfo[0].sampler = defaultSampler.handle;

				desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				desc[0].dstSet = quadTexDescriptor;
				desc[0].dstBinding = 0;
				desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				desc[0].descriptorCount = 1;
				desc[0].pImageInfo = &imageInfo[0];


				constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
				vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
			}

			{
				VkWriteDescriptorSet desc[1]{};
				VkDescriptorImageInfo imageInfo[1]{};
				imageInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				imageInfo[0].imageView = intermediateBrightColorView.handle;
				imageInfo[0].sampler = defaultSampler.handle;

				desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				desc[0].dstSet = brightColorDesc;
				desc[0].dstBinding = 0;
				desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				desc[0].descriptorCount = 1;
				desc[0].pImageInfo = &imageInfo[0];

				constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
				vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
			}

			{
				VkWriteDescriptorSet desc[1]{};
				VkDescriptorImageInfo imageInfo[1]{};
				imageInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				imageInfo[0].imageView = bloomHorizontalView.handle;
				imageInfo[0].sampler = defaultSampler.handle;

				desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				desc[0].dstSet = bloomHorizontalDesc;
				desc[0].dstBinding = 0;
				desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				desc[0].descriptorCount = 1;
				desc[0].pImageInfo = &imageInfo[0];

				constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
				vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
			}


			{
				VkWriteDescriptorSet desc[1]{};
				VkDescriptorImageInfo imageInfo[1]{};
				imageInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				imageInfo[0].imageView = bloomVerticalView.handle;
				imageInfo[0].sampler = defaultSampler.handle;

				desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				desc[0].dstSet = bloomVerticalDesc;
				desc[0].dstBinding = 0;
				desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				desc[0].descriptorCount = 1;
				desc[0].pImageInfo = &imageInfo[0];

				constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
				vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
			}

			if (changes.changedSize)
			{
				pipe = create_pipeline(window, renderPass.handle, pipeLayout.handle);
				bloomHorizontalPipe = create_bloom_pipeline(window, bloomHorizontalPass.handle, bloomPipeLayout.handle, cfg::kBloomH_FragShaderPath);
				bloomVerticalPipe = create_bloom_pipeline(window, bloomVerticalPass.handle, bloomPipeLayout.handle, cfg::kBloomV_FragShaderPath);
				postPipe = create_post_processing_pipeline(window, renderPassB.handle, postPipeLayout.handle);
				
			}
			recreateSwapchain = false;
			continue;
		}

		//prepare data for this frame(section 3)
		glsl::SceneUniform sceneUniforms{};
		update_scene_uniforms(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height, state);

		record_commandsA(intermediateCmdBuffer, renderPass.handle, pipe.handle, window.swapchainExtent,sceneUBO.buffer, sceneUniforms, pipeLayout.handle, sceneDescriptors, ourModel, intermediateFramebuffer.handle);
		submit_commandsA(window, intermediateCmdBuffer, intermediateCbfence.handle);

		
		if (auto const res = vkWaitForFences(window.device, 1, &intermediateCbfence.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Waiting for upload to complete\n" "vkWaitForFences() returned %s", lut::to_string(res).c_str());
		}
		if (auto const res = vkResetFences(window.device, 1, &intermediateCbfence.handle); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence \n" "vkResetFences() returned %s", lut::to_string(res).c_str());
		}
		

		//acquire swapchain image.
		//wait for command buffer to be available
		//record and submit commands
		//present rendered images (note: use the present_results() method)
		std::uint32_t imageIndex = 0;
		auto const acquireRes = vkAcquireNextImageKHR(window.device,
			window.swapchain, std::numeric_limits<std::uint64_t>::max(),
			imageAvailable.handle, VK_NULL_HANDLE, &imageIndex);

		if (VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes)
		{
			//This occurs e.g., when the window has been resized. In this case we needs to recreate
			//the swap chain match the new dimensions. Any resources that directly depend on the swap chain
			//need to be recreated as well. While rare, re_creating the swap chain may give us a different 
			//image format, which we should handle.
			//In both cases, we set the falg that the swap chain has to be re-created and jump to the top of 
			// the loop. Technically, with the VK_SUBOPTIMAL_KHR return code, we could continue rendering with the current
			// swapchain (unlike VK_ERROR_OUT_OF_DATA_KHR, which does require us to recreate the swap chain).
			recreateSwapchain = true;
			continue;
		}

		if (VK_SUCCESS != acquireRes)
		{
			throw lut::Error("Unable to acquire next swapchain image\n" "vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str());
		}

		//Make sure that the command buffer is no loger in use
		assert(std::size_t(imageIndex) < cbfences.size());

		if (auto const res = vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n" "vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		if (auto const res = vkResetFences(window.device, 1, &cbfences[imageIndex].handle); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence %u\n" "vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		//Update state
		auto const now = Clock_::now();
		auto const dt = std::chrono::duration_cast<Secondsf_>(now - previousClock).count();
		previousClock = now;

		update_user_state(state, dt);

		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());
		

		record_commandsB(cbuffers[imageIndex], framebuffers[imageIndex].handle, window.swapchainExtent, renderPassB.handle, postPipe.handle, postPipeLayout.handle, quadTexDescriptor,
			bloomHorizontalFramebuffer.handle, bloomVerticalFramebuffer.handle, bloomHorizontalPass.handle, bloomVerticalPass.handle, bloomPipeLayout.handle, bloomHorizontalPipe.handle,
			bloomVerticalPipe.handle, brightColorDesc, bloomHorizontalDesc, bloomVerticalDesc, queryPool);

		submit_commandsB(window, cbuffers[imageIndex], cbfences[imageIndex].handle, imageAvailable.handle, renderFinished.handle /*, renderPassAComplete.handle*/ );

		present_results(window.presentQueue, window.swapchain, imageIndex, renderFinished.handle, recreateSwapchain);

		cfg::frameCounter++;
		if (cfg::frameCounter % 10 == 0)// Update every 10 frames
		{
			// Wait for the device to become idle to ensure that the queries have finished executing
			vkDeviceWaitIdle(window.device);

			// Retrieve the timestamps from the query pool and calculate the time difference
			uint64_t timestamps[2];
			vkGetQueryPoolResults(window.device, queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

			uint64_t duration = timestamps[1] - timestamps[0];
			VkPhysicalDeviceProperties deviceProperties;
			vkGetPhysicalDeviceProperties(window.physicalDevice, &deviceProperties);

			float timeInMilliseconds = static_cast<float>(duration) * deviceProperties.limits.timestampPeriod * 1e-6f;
			std::cout<< "Bloom performance: " << timeInMilliseconds << " ms" << std::endl;
		}

	}

	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.
	vkDeviceWaitIdle(window.device);

	if (queryPool != VK_NULL_HANDLE) { 
		vkDestroyQueryPool(window.device, queryPool, nullptr); 
		queryPool = VK_NULL_HANDLE; 
	}

	return 0;
}
catch (std::exception const& eErr)
{
	std::fprintf(stderr, "\n");
	std::fprintf(stderr, "Error: %s\n", eErr.what());
	return 1;
}

namespace
{
	void glfw_callback_key_press(GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/)
	{
		if (GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction)
		{
			glfwSetWindowShouldClose(aWindow, GLFW_TRUE);
		}

		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
		assert(state);

		bool const isReleased = (GLFW_RELEASE == aAction);

		switch (aKey)
		{
		case GLFW_KEY_W:
			state->inputMap[std::size_t(EInputState::forward)] = !isReleased;
			break;
		case GLFW_KEY_S:
			state->inputMap[std::size_t(EInputState::backward)] = !isReleased;
			break;
		case GLFW_KEY_A:
			state->inputMap[std::size_t(EInputState::strafeLeft)] = !isReleased;
			break;
		case GLFW_KEY_D:
			state->inputMap[std::size_t(EInputState::strafeRight)] = !isReleased;
			break;
		case GLFW_KEY_E:
			state->inputMap[std::size_t(EInputState::levitate)] = !isReleased;
			break;
		case GLFW_KEY_Q:
			state->inputMap[std::size_t(EInputState::sink)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_SHIFT: [[fallthrough]];
		case GLFW_KEY_RIGHT_SHIFT:
			state->inputMap[std::size_t(EInputState::fast)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_CONTROL: [[fallthrough]];
		case GLFW_KEY_RIGHT_CONTROL:
			state->inputMap[std::size_t(EInputState::slow)] = !isReleased;
			break;

		case GLFW_KEY_SPACE:
			if (aAction == GLFW_PRESS)
			{
				state->inputMap[std::size_t(EInputState::lightRotate)] = !state->inputMap[std::size_t(EInputState::lightRotate)];
			}
			break;

		default:
			;
		}

	}

	void glfw_callback_button(GLFWwindow* aWin, int aBut, int aAct, int)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		if (GLFW_MOUSE_BUTTON_RIGHT == aBut && GLFW_PRESS == aAct)
		{
			auto& flag = state->inputMap[std::size_t(EInputState::mousing)];

			flag = !flag;
			if (flag)
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			else
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}

	void glfw_callback_motion(GLFWwindow* aWin, double aX, double aY)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		state->mouseX = float(aX);
		state->mouseY = float(aY);
	}
}

namespace
{
	void update_scene_uniforms(glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight, UserState aState)
	{
		//TODO- (Section 3) initialize SceneUniform members
		float const aspect = aFramebufferWidth / float(aFramebufferHeight);

		aSceneUniforms.projection = glm::perspectiveRH_ZO(lut::Radians(cfg::kCameraFov).value(),
			aspect, cfg::kCameraNear, cfg::kCameraFar);
		aSceneUniforms.projection[1][1] *= -1.f;// mirror Y axis

		aSceneUniforms.camera = glm::inverse(aState.camera2world);

		aSceneUniforms.projCam = aSceneUniforms.projection * aSceneUniforms.camera;
		glm::translate(aState.camera2world, glm::vec3(0, 2, -5));
		aSceneUniforms.cameraPos = glm::vec3(aState.camera2world[3]);

		aSceneUniforms.lightColor[0] = glm::vec4(2, 1, 1, 1);//strong white
		aSceneUniforms.lightColor[1] = glm::vec4(10, 0.5, 0, 1);//very strong orange
		aSceneUniforms.lightColor[2] = glm::vec4(0, 0.1, 1.5, 1);//weak blue
		aSceneUniforms.lightPos[0] = aState.light_pos0;
		aSceneUniforms.lightPos[1] = aState.light_pos1;
		aSceneUniforms.lightPos[2] = aState.light_pos2;
	}
}

namespace
{
	void update_user_state(UserState& aState, float aElapsedTime)
	{
		auto& cam = aState.camera2world;

		if (aState.inputMap[std::size_t(EInputState::mousing)])
		{

			if (aState.wasMousing)
			{
				auto const sens = cfg::kCameraMouseSensitivity;
				auto const dx = sens * (aState.mouseX - aState.previousX);
				auto const dy = sens * (aState.mouseY - aState.previousY);

				cam *= glm::rotate(-dy, glm::vec3(1.f, 0.f, 0.f));
				cam *= glm::rotate(-dx, glm::vec3(0.f, 1.f, 0.f));
			}

			aState.previousX = aState.mouseX;
			aState.previousY = aState.mouseY;
			aState.wasMousing = true;
		}
		else
		{
			aState.wasMousing = false;
		}

		if (aState.inputMap[std::size_t(EInputState::lightRotate)])
		{
			glm::mat4 rotationMatrix0 = glm::rotate(glm::mat4(1.0f), glm::radians(45.0f) * aElapsedTime, glm::vec3(0.0f, 1.0f, 0.0f));
			glm::vec4 center0(0.0f, 0.0f, -2.0f, 1.0f);
			aState.light_pos0 = rotationMatrix0 * (aState.light_pos0 - center0) + center0;

			glm::mat4 rotationMatrix1 = glm::rotate(glm::mat4(1.0f), glm::radians(30.0f) * aElapsedTime, glm::vec3(1.0f, 0.0f, 0.0f));
			glm::vec4 center1(0.0f, 2.0f, 0.0f, 1.0f);
			aState.light_pos1 = rotationMatrix1 * (aState.light_pos1 - center1) + center1;

			glm::mat4 rotationMatrix2 = glm::rotate(glm::mat4(1.0f), glm::radians(60.0f) * aElapsedTime, glm::vec3(0.0f, 0.0f, 1.0f));
			glm::vec4 center2(0.0f, 2.0f, 0.0f, 1.0f);
			aState.light_pos2 = rotationMatrix2 * (aState.light_pos2 - center2) + center2;
		}

		auto const move = aElapsedTime * cfg::kCameraBaseSpeed *
			(aState.inputMap[std::size_t(EInputState::fast)] ? cfg::kCameraFastMult : 1.f) *
			(aState.inputMap[std::size_t(EInputState::slow)] ? cfg::kCameraSlowMult : 1.f);

		if (aState.inputMap[std::size_t(EInputState::forward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, -move));
		if (aState.inputMap[std::size_t(EInputState::backward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, +move));

		if (aState.inputMap[std::size_t(EInputState::strafeLeft)])
			cam = cam * glm::translate(glm::vec3(-move, 0.f, 0.f));
		if (aState.inputMap[std::size_t(EInputState::strafeRight)])
			cam = cam * glm::translate(glm::vec3(+move, 0.f, 0.f));

		if (aState.inputMap[std::size_t(EInputState::levitate)])
			cam = cam * glm::translate(glm::vec3(0.f, +move, 0.f));
		if (aState.inputMap[std::size_t(EInputState::sink)])
			cam = cam * glm::translate(glm::vec3(0.f, -move, 0.f));

	}

	lut::RenderPass create_render_pass_A(lut::VulkanWindow const& aWindow)
	{
		//TODO- (Section 1 / Exercise 3) implement me!
		VkAttachmentDescription attachments[3]{};
		attachments[0].format = VK_FORMAT_R16G16B16A16_SFLOAT;  //for HDR
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		attachments[1].format = VK_FORMAT_R16G16B16A16_SFLOAT;  //for HDR
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		attachments[2].format = cfg::kDepthFormat;
		attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[2].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference subpassAttachments[2]{};
		subpassAttachments[0].attachment = 0; //this refers to attachment[0]
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		subpassAttachments[1].attachment = 1; //this refers to attachment[1]
		subpassAttachments[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 2; //this refers to attachments[0]
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 2;
		subpasses[0].pColorAttachments = subpassAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachment;

		
		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 3;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		//passInfo.dependencyCount = 2;  
		//passInfo.pDependencies = dependencies;  
		passInfo.dependencyCount = 0;
		passInfo.pDependencies = nullptr;
		
		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n" "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}
		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::RenderPass create_render_pass_B(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[1]{};
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE; 
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; 
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0; //this refers to attachment[0]
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;


		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; 
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;

		//changed: no explicit subpass dependencies
		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 1;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 0;  
		passInfo.pDependencies = nullptr; 

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n" "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}
		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::RenderPass create_render_pass_Bloom(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[1]{};

		attachments[0].format = VK_FORMAT_R16G16B16A16_SFLOAT;  //for HDR
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0;
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;


		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = &subpassAttachments[0];

		
		VkSubpassDependency dependencies[2]{}; //////////////////////////////////////comeback later
		
		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		
		//for more render passes performance testing
		dependencies[1].srcSubpass = 0; 
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL; 
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; 
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT; 
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; 
		dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT; 
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT; 

		
		
		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 1;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 2;
		passInfo.pDependencies = dependencies;


		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n" "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}
		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout aSceneLayout, VkDescriptorSetLayout aObjectLayout)
	{
		//TODO- (Section 1 / Exercise 3) implement me!
		VkDescriptorSetLayout layouts[] =
		{  //Order must match the set = N in the shaders
			aSceneLayout ,//set 0
			aObjectLayout //set1
		};

		//TODO: implement me!
		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n""vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::PipelineLayout create_post_processing_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout aTexLayout)
	{
		VkDescriptorSetLayout layouts[] =
		{  //Order must match the set = N in the shaders
			aTexLayout//set 0
		};

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n""vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::PipelineLayout create_final_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout aTexLayout, VkDescriptorSetLayout aBloomVLayout)
	{
		VkDescriptorSetLayout layouts[] =
		{  //Order must match the set = N in the shaders
			aTexLayout ,
			aBloomVLayout//set 0
		};

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n""vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::PipelineLayout(aContext.device, layout);
	}


	lut::Pipeline create_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFragShaderPath);

		//Define shader stages in the pipeline
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		//define depth and stencil state
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;


		VkVertexInputBindingDescription vertexInputs[1]{};
		//textured meshes
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 8;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttribs[3]{};

		//positions
		vertexAttribs[0].binding = 0; // must match binding above
		vertexAttribs[0].location = 0; // must match shader
		vertexAttribs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttribs[0].offset = 0;

		////texcoords
		vertexAttribs[1].binding = 0;
		vertexAttribs[1].location = 1;
		vertexAttribs[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttribs[1].offset = sizeof(float) * 3;

		//normals
		vertexAttribs[2].binding = 0;
		vertexAttribs[2].location = 2;
		vertexAttribs[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttribs[2].offset = sizeof(float) * 5;


		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 1;
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 3;
		inputInfo.pVertexAttributeDescriptions = vertexAttribs;

		// Define which primitive (point, line, triangle, ...) the input is
		// assembled into for rasterization. 
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		//Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = static_cast<float>(aWindow.swapchainExtent.width);
		viewport.height = static_cast<float>(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = aWindow.swapchainExtent;

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//Define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f;

		//Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Define blend state

		VkPipelineColorBlendAttachmentState blendStates[2]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		blendStates[1].blendEnable = VK_FALSE;
		blendStates[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 2;
		blendInfo.pAttachments = blendStates;

		//Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2; // vert + frag stages
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;  // no tesselation
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;   // no dynamic states

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;  // first subpass of aRenderPass

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n" "vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_post_processing_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kFullScreenVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFullScreenFragShaderPath);

		//Define shader stages in the pipeline
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";


		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;


		// Define which primitive (point, line, triangle, ...) the input is
		// assembled into for rasterization. 
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		//Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = static_cast<float>(aWindow.swapchainExtent.width);
		viewport.height = static_cast<float>(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = aWindow.swapchainExtent;

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//Define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		//rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f;

		//Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Define blend state
		// We define one blend state per color attachment - this example uses a
		// single color attachment, so we only need one. Right now, we don¡¯t do any
		// blending, so we can ignore most of the members.
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		//Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2; // vert + frag stages
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;  // no tesselation
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;   // no dynamic states

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;  // first subpass of aRenderPass

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n" "vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_bloom_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout, const char* FragShaderPath)
	{
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kFullScreenVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, FragShaderPath);

		//Define shader stages in the pipeline
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";


		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;


		// Define which primitive (point, line, triangle, ...) the input is
		// assembled into for rasterization. 
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		//Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = static_cast<float>(aWindow.swapchainExtent.width);
		viewport.height = static_cast<float>(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = aWindow.swapchainExtent;

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//Define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f;

		//Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		//Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2; // vert + frag stages
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;  // no tesselation
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;   // no dynamic states

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;  // first subpass of aRenderPass

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n" "vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}



	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imgInfo{};
		imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imgInfo.imageType = VK_IMAGE_TYPE_2D;
		imgInfo.format = cfg::kDepthFormat;
		imgInfo.extent.width = aWindow.swapchainExtent.width;
		imgInfo.extent.height = aWindow.swapchainExtent.height;
		imgInfo.extent.depth = 1;
		imgInfo.mipLevels = 1;
		imgInfo.arrayLayers = 1;
		imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imgInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (auto const res = vmaCreateImage(aAllocator.allocator, &imgInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to allocate depth buffer image.\n" "vmaCreateImage() returned %s", lut::to_string(res).c_str());
		}

		lut::Image depthImage(aAllocator.allocator, image, allocation);

		//create the image view
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{}; //identity
		viewInfo.subresourceRange = VkImageSubresourceRange{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

		VkImageView view = VK_NULL_HANDLE;
		if (auto const res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n" "vkCreateImageView() returned %s", lut::to_string(res).c_str());
		}

		return { std::move(depthImage), lut::ImageView(aWindow.device, view) };
	}

	void create_swapchain_framebuffers(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers)
	{
		assert(aFramebuffers.empty());

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i)
		{
			VkImageView attachments[1] = { aWindow.swapViews[i]};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0;
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 1;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;
			VkFramebuffer fb = VK_NULL_HANDLE;
			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create framebuffer for swap chain image %zu\n" "vkCreateFramebuffer returned %s", i, lut::to_string(res).c_str());
			}

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}
		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}

	void create_Intermediate_framebuffers(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, lut::Framebuffer& aFramebuffers, VkImageView aColorView, VkImageView aBrightColorView, VkImageView aDepthView)
	{
		VkImageView attachments[3] = { aColorView,aBrightColorView, aDepthView };

		VkFramebufferCreateInfo fbInfo{};
		fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbInfo.flags = 0;
		fbInfo.renderPass = aRenderPass;
		fbInfo.attachmentCount = 3;
		fbInfo.pAttachments = attachments;
		fbInfo.width = aWindow.swapchainExtent.width;
		fbInfo.height = aWindow.swapchainExtent.height;
		fbInfo.layers = 1;
		VkFramebuffer fb = VK_NULL_HANDLE;
		if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create framebuffer for RenderPassB image\n" "vkCreateFramebuffer returned %s", lut::to_string(res).c_str());
		}

		aFramebuffers = std::move(lut::Framebuffer(aWindow.device, fb));
	}

	void create_bloom_framebuffer(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, lut::Framebuffer& aFramebuffers, VkImageView aView)
	{
		VkImageView attachments[1] = { aView};

		VkFramebufferCreateInfo fbInfo{}; 
		fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO; 
		fbInfo.flags = 0; 
		fbInfo.renderPass = aRenderPass; 
		fbInfo.attachmentCount = 1;
		fbInfo.pAttachments = attachments; 
		fbInfo.width = aWindow.swapchainExtent.width; 
		fbInfo.height = aWindow.swapchainExtent.height; 
		fbInfo.layers = 1; 
		VkFramebuffer fb = VK_NULL_HANDLE; 
		if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res) 
		{
			throw lut::Error("Unable to create framebuffer for RenderPassB image\n" "vkCreateFramebuffer returned %s", lut::to_string(res).c_str()); 
		}

		aFramebuffers = std::move(lut::Framebuffer(aWindow.device, fb));
	}  


	void record_commandsA(VkCommandBuffer aCmdBuff, VkRenderPass aRenderPassA,
		VkPipeline aGraphicsPipe, VkExtent2D const& aImageExtent, VkBuffer aSceneUBO, glsl::SceneUniform const& aSceneUniform,
		VkPipelineLayout aGraphicsLayout, VkDescriptorSet aSceneDescriptors, ModelPack& aModel, VkFramebuffer aIntmdtFramebuffer)
	{
		//Begin recording commands
		VkCommandBufferBeginInfo begInfo{};
		begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		begInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &begInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n" "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		//Upload scene uniforms
		lut::buffer_barrier(aCmdBuff, aSceneUBO, VK_ACCESS_UNIFORM_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

		vkCmdUpdateBuffer(aCmdBuff, aSceneUBO, 0, sizeof(glsl::SceneUniform), &aSceneUniform);

		lut::buffer_barrier(aCmdBuff, aSceneUBO, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
		for (auto& mesh : aModel.meshes)
		{
			//Upload tex uniforms
			lut::buffer_barrier(aCmdBuff, aModel.texUBOs[mesh.matID].buffer, VK_ACCESS_UNIFORM_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

			vkCmdUpdateBuffer(aCmdBuff, aModel.texUBOs[mesh.matID].buffer, 0, sizeof(TexParameter), &aModel.texParameters[mesh.matID]);

			lut::buffer_barrier(aCmdBuff, aModel.texUBOs[mesh.matID].buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
		}

		//Begin render pass
		VkClearValue clearValues[3]{};
		clearValues[0].color.float32[0] = 0.1f;
		clearValues[0].color.float32[1] = 0.1f;
		clearValues[0].color.float32[2] = 0.1f;
		clearValues[0].color.float32[3] = 1.f;

		clearValues[1].color.float32[0] = 0.0f;
		clearValues[1].color.float32[1] = 0.0f;
		clearValues[1].color.float32[2] = 0.0f;
		clearValues[1].color.float32[3] = 1.f;

		clearValues[2].depthStencil.depth = 1.f;

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = aRenderPassA;
		passInfo.framebuffer = aIntmdtFramebuffer; // changeLater
		passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo.renderArea.extent = aImageExtent;
		passInfo.clearValueCount = 3;
		passInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

		//Begin drawing with our graphics pipeline
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsPipe);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 0, 1, &aSceneDescriptors, 0, nullptr);
		for (auto& mesh : aModel.meshes)
		{
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 1, 1, &aModel.matDecriptors[mesh.matID], 0, nullptr);
			VkDeviceSize offsets[1]{};
			vkCmdBindVertexBuffers(aCmdBuff, 0, 1, &mesh.vertices.buffer, offsets);
			vkCmdBindIndexBuffer(aCmdBuff, mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(aCmdBuff, mesh.indexCount, 1, 0, 0, 0);

		}

		vkCmdEndRenderPass(aCmdBuff);
		

		//End command recording
		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res)
			throw lut::Error("Unable to end recording command buffer\n" "vkEndCoomandBuffer{} returned %s", lut::to_string(res).c_str());
	}

	void record_commandsB(VkCommandBuffer aCmdBuff, VkFramebuffer aFramebuffer, VkExtent2D const& aImageExtent, 
		VkRenderPass aRenderPassB, VkPipeline aPostPipe,VkPipelineLayout aPostPipeLayout, VkDescriptorSet aFullscreenDesc,
		VkFramebuffer aBloomHFramebuffer, VkFramebuffer aBloomVFramebuffer, VkRenderPass aBloomHPass, VkRenderPass aBloomVPass, VkPipelineLayout aBloomPipeLayout,
		VkPipeline aBloomHPipe, VkPipeline aBloomVPipe, VkDescriptorSet aBrightColorDesc, VkDescriptorSet aBloomHDesc, VkDescriptorSet aBloomVDesc, VkQueryPool aQueryPool)
	{
		//Begin recording commands
		VkCommandBufferBeginInfo begInfo{};
		begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		begInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &begInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n" "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		//bloom horizontal renderPass 
		{
			VkClearValue clearValues[1]{};
			clearValues[0].color.float32[0] = 0.0f;
			clearValues[0].color.float32[1] = 0.0f;
			clearValues[0].color.float32[2] = 0.0f;
			clearValues[0].color.float32[3] = 1.f;


			VkRenderPassBeginInfo passInfo{};
			passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			passInfo.renderPass = aBloomHPass;
			passInfo.framebuffer = aBloomHFramebuffer;
			passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
			passInfo.renderArea.extent = aImageExtent;
			passInfo.clearValueCount = 1;
			passInfo.pClearValues = clearValues;

			vkCmdResetQueryPool(aCmdBuff, aQueryPool, 0, 2);

			vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdWriteTimestamp(aCmdBuff, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, aQueryPool, 0);

			//Begin drawing with our graphics pipeline
			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aBloomHPipe);
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aBloomPipeLayout, 0, 1, &aBrightColorDesc, 0, nullptr);

			vkCmdDraw(aCmdBuff, 3, 1, 0, 0);

			vkCmdEndRenderPass(aCmdBuff);
		}

		//bloom vertical renderPass
		{
			VkClearValue clearValues[1]{};
			clearValues[0].color.float32[0] = 0.0f;
			clearValues[0].color.float32[1] = 0.0f;
			clearValues[0].color.float32[2] = 0.0f;
			clearValues[0].color.float32[3] = 1.f;


			VkRenderPassBeginInfo passInfo{};
			passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			passInfo.renderPass = aBloomVPass; 
			passInfo.framebuffer = aBloomVFramebuffer;
			passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
			passInfo.renderArea.extent = aImageExtent;
			passInfo.clearValueCount = 1;
			passInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

			//Begin drawing with our graphics pipeline
			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aBloomVPipe);
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aBloomPipeLayout, 0, 1, &aBloomHDesc, 0, nullptr); 

			vkCmdDraw(aCmdBuff,3, 1, 0, 0);
			//vkCmdWriteTimestamp(aCmdBuff, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, aQueryPool, 1);
			vkCmdEndRenderPass(aCmdBuff);
		}

		
		//this is for testing performance
		
		for (int i = 1; i < 2; ++i)
		{
			//bloom horizontal renderPass 
			{
				VkClearValue clearValues[1]{};
				clearValues[0].color.float32[0] = 0.0f;
				clearValues[0].color.float32[1] = 0.0f;
				clearValues[0].color.float32[2] = 0.0f;
				clearValues[0].color.float32[3] = 1.f;


				VkRenderPassBeginInfo passInfo{};
				passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
				passInfo.renderPass = aBloomHPass;
				passInfo.framebuffer = aBloomHFramebuffer;
				passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
				passInfo.renderArea.extent = aImageExtent;
				passInfo.clearValueCount = 1;
				passInfo.pClearValues = clearValues;

				vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

				//Begin drawing with our graphics pipeline
				vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aBloomHPipe);
				vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aBloomPipeLayout, 0, 1, &aBloomVDesc, 0, nullptr);

				vkCmdDraw(aCmdBuff, 3, 1, 0, 0);

				vkCmdEndRenderPass(aCmdBuff);
			}

			//bloom vertical renderPass
			{
				VkClearValue clearValues[1]{};
				clearValues[0].color.float32[0] = 0.0f;
				clearValues[0].color.float32[1] = 0.0f;
				clearValues[0].color.float32[2] = 0.0f;
				clearValues[0].color.float32[3] = 1.f;


				VkRenderPassBeginInfo passInfo{};
				passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
				passInfo.renderPass = aBloomVPass;
				passInfo.framebuffer = aBloomVFramebuffer;
				passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
				passInfo.renderArea.extent = aImageExtent;
				passInfo.clearValueCount = 1;
				passInfo.pClearValues = clearValues;

				vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

				//Begin drawing with our graphics pipeline
				vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aBloomVPipe);
				vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aBloomPipeLayout, 0, 1, &aBloomHDesc, 0, nullptr);

				vkCmdDraw(aCmdBuff, 3, 1, 0, 0);
				if(i == 1)
				vkCmdWriteTimestamp(aCmdBuff, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, aQueryPool, 1);
				vkCmdEndRenderPass(aCmdBuff);
			}
		}
		
		
		
		//------------------------------------------------------------------
		//Begin render passB
		{
			VkClearValue clearValues[1]{};
			clearValues[0].color.float32[0] = 0.0f;
			clearValues[0].color.float32[1] = 0.0f;
			clearValues[0].color.float32[2] = 0.0f;
			clearValues[0].color.float32[3] = 1.f;


			VkRenderPassBeginInfo passInfo{};
			passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			passInfo.renderPass = aRenderPassB;
			passInfo.framebuffer = aFramebuffer;
			passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
			passInfo.renderArea.extent = aImageExtent;
			passInfo.clearValueCount = 1;
			passInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

			//Begin drawing with our graphics pipeline
			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPostPipe);
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPostPipeLayout, 0, 1, &aFullscreenDesc, 0, nullptr);
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPostPipeLayout, 1, 1, &aBloomVDesc, 0, nullptr);

			vkCmdDraw(aCmdBuff, 3, 1, 0, 0);

			vkCmdEndRenderPass(aCmdBuff);
		}

		//End command recording
		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res)
			throw lut::Error("Unable to end recording command buffer\n" "vkEndCoomandBuffer{} returned %s", lut::to_string(res).c_str());
	}

	void submit_commandsA(lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence)
	{

		VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;

		//submitInfo.waitSemaphoreCount = 1;
		//submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = &waitPipelineStages;

		//submitInfo.signalSemaphoreCount = 1;
		//submitInfo.pSignalSemaphores = &aSignalSemaphore;

		if (auto const res = vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, aFence); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to submit command buffer to queue\n" "vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}
	}

	void submit_commandsB(lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore/*, VkSemaphore aWaitSemaphore2*/)
	{

		VkPipelineStageFlags waitPipelineStages[2] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT ,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;

		//VkSemaphore waitSemaphores[2] = { aWaitSemaphore, aWaitSemaphore2 };
		//submitInfo.waitSemaphoreCount = 2;
		//submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = waitPipelineStages;

		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;

		if (auto const res = vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, aFence); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to submit command buffer to queue\n" "vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}
	}

	void present_results(VkQueue aPresentQueue, VkSwapchainKHR aSwapchain, std::uint32_t aImageIndex, VkSemaphore aRenderFinished, bool& aNeedToRecreateSwapchain)
	{
		//TODO: (Section 1/Exercise 3) implement me!
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &aRenderFinished;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &aSwapchain;
		presentInfo.pImageIndices = &aImageIndex;
		presentInfo.pResults = nullptr;
		auto const presentRes = vkQueuePresentKHR(aPresentQueue, &presentInfo);
		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
		{
			aNeedToRecreateSwapchain = true;
		}
		else if (VK_SUCCESS != presentRes)
		{
			throw lut::Error("Unable present swapchain image %u\n"
				"vkQueuePresentKHR() returned %s", aImageIndex, lut::to_string(presentRes).
				c_str());
		}
	}


	lut::DescriptorSetLayout create_material_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[4]{};

		// basecolor
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		// roughness
		bindings[1].binding = 1;
		bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[1].descriptorCount = 1;
		bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		// metalness
		bindings[2].binding = 2;
		bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[2].descriptorCount = 1;
		bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		//parameters UBO
		bindings[3].binding = 3;
		bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; 
		bindings[3].descriptorCount = 1;
		bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};
		layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutCreateInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutCreateInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;

		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutCreateInfo, nullptr, &layout); VK_SUCCESS != res) {
			throw lut::Error("Unable to create decriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_fullScreen_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[2]{};

		// intermediate texture
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[0].descriptorCount = 1; 
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; 

		bindings[1].binding = 1;
		bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[1].descriptorCount = 1;
		bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;


		VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};
		layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutCreateInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutCreateInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;

		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutCreateInfo, nullptr, &layout); VK_SUCCESS != res) {
			throw lut::Error("Unable to create decriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_bloom_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};

		// intermediate texture
		bindings[0].binding = 0; 
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; 
		bindings[0].descriptorCount = 1; 
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutCreateInfo{}; 
		layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; 
		layoutCreateInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutCreateInfo.pBindings = bindings; 

		VkDescriptorSetLayout layout = VK_NULL_HANDLE; 

		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutCreateInfo, nullptr, &layout); VK_SUCCESS != res) {
			throw lut::Error("Unable to create decriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; // number must match the index of the corresponding binding = N declaration in the shader(s)

		bindings[0].descriptorCount = 1;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create decriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
		}
		return lut::DescriptorSetLayout(aWindow.device, layout);

	}
}




//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
