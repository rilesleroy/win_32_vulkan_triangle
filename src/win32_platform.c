// TODO(shmoich): time step
// TODO(shmoich): logging
// TODO(shmoich): input
// TODO(shmoich): implement strdup (that wierd crashing was do to me doing shitt string manipulation)
// TODO(shmoich): pull needed vulkan vars out into there own struct
///// vulkan stuff
// TODO(shmoich): I need a reasonable way to do get the window width and height
// TODO(shmoich): I need some sort of perminate storage for the swap chain images

// Platform specific
#include <windows.h>
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

// C std
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>

#define globalvar static
#define internal static

globalvar bool global_running;

typedef union {
  struct {
    float x, y;
  };
  struct {
    float u, v;
  };
  float elements[2];
} vec2_t;

typedef union {
  struct {
    float x, y, z;
  };
  struct{
    float r, g, b;
  };
  struct {
    float u, v, w;
  };
  float elements[3];
} vec3_t;





/////////////////////
// Min Max Clamp
#define Min(v, l) ((v < l) ? (v) : (l))
#define Max(v, h) ((v > h) ? (v) : (h))
#define Clamp(v, l, h) (Min(h, Min(v, l)))

/////////////////////
// Super simple scratch/arena allocator
// not memory alligned
typedef struct {
  size_t total;
  size_t offset;
  uint8_t* buffer;
} Scratch;

Scratch ScratchInit(size_t total)
{
  Scratch scratch;
  scratch.total = total;
  scratch.offset = 0;
  scratch.buffer = (uint8_t*) malloc(total);
  memset((void *)(scratch.buffer), 0, total);
  return scratch;
}

void* ScratchAlloc(Scratch* scratch, size_t size)
{
  if(size + scratch->offset < scratch->total)
  {
    void* ptr = (void*)(scratch->buffer + scratch->offset);
    scratch->offset += size;
    return ptr;
  }
  return 0;
}

void ScratchFreeAll(Scratch* scratch)
{
  memset((void *)(scratch->buffer), 0, scratch->total);
  scratch->offset = 0;
}

void ScratchDestroy(Scratch* scratch)
{
  free(scratch->buffer);
}

// TODO(shmoic): replace this with a better solution later
globalvar Scratch permanent_storage;

// Vertex
typedef struct {
  vec2_t pos;
  vec3_t color;
} vertex_t;

globalvar vertex_t verts[3] = {
  {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
  {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
  {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
};

VkVertexInputBindingDescription get_binding_description()
{
  VkVertexInputBindingDescription bindingDescription = {
    .binding = 0,
    .stride = sizeof(vertex_t),
    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
  };
  return bindingDescription;
}

VkVertexInputAttributeDescription* get_attribute_descriptions()
{
  VkVertexInputAttributeDescription* attribute_description = ScratchAlloc(&permanent_storage, sizeof(VkVertexInputAttributeDescription) * 2);
  
  attribute_description[0].binding = 0;
  attribute_description[0].location = 0;
  attribute_description[0].format = VK_FORMAT_R32G32_SFLOAT;
  attribute_description[0].offset = offsetof(vertex_t, pos);
  
  attribute_description[1].binding = 0;
  attribute_description[1].location = 1;
  attribute_description[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attribute_description[1].offset = offsetof(vertex_t, color);
  
  
  return attribute_description;
}

/////////////////
// Win32 resizing 
typedef struct
{
  int32_t height;
  int32_t width;
} WindowDimension;

// TODO(shmoic): this should gow into the renderer state at some point
globalvar WindowDimension global_window_dims;
WindowDimension Win32_GetWindowDims(HWND window)
{
  WindowDimension new_dims;
  RECT rectangle;
  GetWindowRect(window, &rectangle);
  new_dims.width = (int32_t) (rectangle.right - rectangle.left);
  new_dims.height = (int32_t) (rectangle.bottom - rectangle.top);
  return new_dims;
}

////////////////
// Vulkan
// TODO(shmoic): this could be a problem at somepoint initiallizing to max can be bad
#define VK_NULL_ALLOC NULL
#define VK_MAX_FRAMES_IN_FLIGHT 2
#define VK_MAX_QUEUE_FAMILY_INDEX 0xffffffff
typedef struct {
  VkInstance instance;
  // if validation layers
  VkDebugUtilsMessengerEXT debug_messenger;
  
  VkSurfaceKHR surface;
  VkPhysicalDevice physical_device;
  VkDevice logical_device; //change to device?
  uint32_t present_queue_family_index;
  VkQueue present_queue;
  uint32_t graphics_queue_family_index;
  VkQueue graphics_queue;
  
  // Swap chain stuff
  uint32_t frame_buffer_count;
  VkSwapchainKHR swap_chain;
  VkExtent2D swap_chain_extent;
  VkFormat swap_chain_format;
  VkImage* swap_chain_images;
  VkImageView* swap_chain_image_views;
  VkFramebuffer* frame_buffers;
  
  // Render Pipeline
  VkRenderPass render_pass;
  VkPipelineLayout pipeline_layout;
  VkPipeline graphics_pipeline;
  
  // Commands
  VkCommandPool command_pool;
  VkCommandBuffer* command_buffers;
  
  // Render/Present
  VkSemaphore* image_available_semaphores;
  VkSemaphore* render_finished_semaphores;
  VkFence* in_flight_fences;
  VkFence* images_in_flight;
  
  size_t current_frame;
  
} VulkanContext;

typedef struct {
  size_t size;
  void* buffer;
}LoadedFile;

void DEBUG_Win32_FreeFileMemory(void *memory)
{
  if(memory)
  {
    free(memory);
  }
}

LoadedFile DEBUG_Win32_ReadEntireFile(char* file_path)
{
  LoadedFile file = {0};
  HANDLE file_handle = CreateFileA(file_path,
                                   GENERIC_READ,
                                   FILE_SHARE_READ,
                                   NULL,
                                   OPEN_EXISTING,
                                   //FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
                                   0,
                                   NULL
                                   );
  
  if (file_handle != INVALID_HANDLE_VALUE)
  {
    LARGE_INTEGER file_size;
    
    if(GetFileSizeEx(file_handle, &file_size))
    {
      file.buffer = malloc(file_size.QuadPart); // use something else / replace?
      file.size = ((size_t)file_size.QuadPart);
      if(file.buffer)
      {
        DWORD bytes_read;
        if(ReadFile(file_handle, file.buffer, (uint32_t)file_size.QuadPart, &bytes_read, NULL) && (bytes_read == (uint32_t)file_size.QuadPart))
        {
          // read successful log read?
        }
        else
        {
          DEBUG_Win32_FreeFileMemory(file.buffer);
          file.buffer = 0;
          file.size = 0;
        }
      }
    }
    CloseHandle(file_handle);
  }
  else
  {
    printf("ERROR: could not open file: %s\n", file_path);
    assert(0);
  }
  
  return file;
}


static VKAPI_ATTR VkBool32 VKAPI_CALL DEBUG_Vulkan_ValidationLayerCallBack (VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                                                                            VkDebugUtilsMessageTypeFlagsEXT message_type,
                                                                            const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                                                                            void* pUserData)
{
  if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    printf("valiodation layer: %s\n", callback_data->pMessage);
  }
  return VK_FALSE;
}

VkResult DEBUG_Vulkan_CreateCallback(VkInstance instance,
                                     const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                     const VkAllocationCallbacks* pAllocator,
                                     VkDebugUtilsMessengerEXT* pDebugMessenger)
{
  PFN_vkCreateDebugUtilsMessengerEXT func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (*func != NULL) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DEBUG_Vulkan_DestroyCallback(VkInstance instance,
                                  VkDebugUtilsMessengerEXT debugMessenger,
                                  VkAllocationCallbacks* pAllocator) 
{
  PFN_vkDestroyDebugUtilsMessengerEXT func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != NULL) {
    func(instance, debugMessenger, pAllocator);
  }
}


void CreateSwapchain(VulkanContext* vulkan_context)
{
  Scratch scratch = ScratchInit(1024 * 100);
  // Swapchain
  uint32_t format_count; 
  vkGetPhysicalDeviceSurfaceFormatsKHR(vulkan_context->physical_device, vulkan_context->surface, &format_count, NULL);
  VkSurfaceFormatKHR* surface_formats = ScratchAlloc(&scratch, sizeof(VkSurfaceFormatKHR) * format_count);
  vkGetPhysicalDeviceSurfaceFormatsKHR(vulkan_context->physical_device, vulkan_context->surface, &format_count, surface_formats);
  
  // Check formats
  VkSurfaceFormatKHR chosen_surface_format;
  bool has_valid_format = false;
  for (size_t i = 0; i < format_count; i++)
  {
    if(surface_formats[i].format == VK_FORMAT_B8G8R8A8_SRGB && surface_formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
    {
      chosen_surface_format = surface_formats[i];
      has_valid_format = true;
      break;
    }
  }
  if (!has_valid_format)
  {
    printf("could not find swap_chain format\n");
    assert(0);
  }
  
  // present_modes
  uint32_t present_mode_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(vulkan_context->physical_device, vulkan_context->surface, &present_mode_count, NULL);
  VkPresentModeKHR* present_modes = ScratchAlloc(&scratch, sizeof(VkPresentModeKHR) * present_mode_count);
  vkGetPhysicalDeviceSurfacePresentModesKHR(vulkan_context->physical_device, vulkan_context->surface, &present_mode_count, present_modes);
  
  VkPresentModeKHR chosen_present_mode = VK_PRESENT_MODE_FIFO_KHR;
  // VK_PRESENT_MODE_MAILBOX_KHR triple buffer?
  // The swap extent is the resolution of the swap chain images
  VkSurfaceCapabilitiesKHR surface_capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vulkan_context->physical_device, vulkan_context->surface, &surface_capabilities);
  
  vulkan_context->swap_chain_extent.width = Clamp(global_window_dims.width, surface_capabilities.minImageExtent.width, surface_capabilities.maxImageExtent.width);
  vulkan_context->swap_chain_extent.height = Clamp(global_window_dims.height, surface_capabilities.minImageExtent.height, surface_capabilities.maxImageExtent.height);
  
  vulkan_context->frame_buffer_count = surface_capabilities.minImageCount + 1;
  
  VkSwapchainCreateInfoKHR swap_chain_create_info;
  swap_chain_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swap_chain_create_info.pNext = NULL;
  swap_chain_create_info.flags = 0;
  swap_chain_create_info.surface = vulkan_context->surface;
  swap_chain_create_info.minImageCount = vulkan_context->frame_buffer_count;
  swap_chain_create_info.imageFormat = chosen_surface_format.format;
  swap_chain_create_info.imageColorSpace = chosen_surface_format.colorSpace;
  swap_chain_create_info.imageExtent = vulkan_context->swap_chain_extent;
  swap_chain_create_info.imageArrayLayers = 1;
  swap_chain_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  swap_chain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  swap_chain_create_info.queueFamilyIndexCount = 0;
  swap_chain_create_info.pQueueFamilyIndices = NULL;
  swap_chain_create_info.preTransform = surface_capabilities.currentTransform;
  swap_chain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swap_chain_create_info.presentMode = chosen_present_mode;
  swap_chain_create_info.clipped = VK_TRUE;
  swap_chain_create_info.oldSwapchain = VK_NULL_HANDLE;
  
  if (vkCreateSwapchainKHR(vulkan_context->logical_device, &swap_chain_create_info, NULL, &vulkan_context->swap_chain) != VK_SUCCESS) 
  {
    printf("could not create swap_chain\n");
    assert(0);
  }
  
  vulkan_context->swap_chain_format = chosen_surface_format.format;
  
  vkGetSwapchainImagesKHR(vulkan_context->logical_device,
                          vulkan_context->swap_chain,
                          &vulkan_context->frame_buffer_count,
                          NULL);
  vulkan_context->swap_chain_images = ScratchAlloc(&permanent_storage,
                                                   sizeof(VkImage) * vulkan_context->frame_buffer_count);
  vkGetSwapchainImagesKHR(vulkan_context->logical_device,
                          vulkan_context->swap_chain,
                          &vulkan_context->frame_buffer_count,
                          vulkan_context->swap_chain_images);
  
  //Image views
  vulkan_context->swap_chain_image_views = ScratchAlloc(&permanent_storage,
                                                        sizeof(VkImageView) * vulkan_context->frame_buffer_count);
  
  for(size_t i=0; i < vulkan_context->frame_buffer_count; i++)
  {
    VkImageViewCreateInfo image_view_create_info = {
      .pNext = NULL,
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = vulkan_context->swap_chain_images[i],
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = vulkan_context->swap_chain_format,
      .components.r = VK_COMPONENT_SWIZZLE_IDENTITY,
      .components.g = VK_COMPONENT_SWIZZLE_IDENTITY,
      .components.b = VK_COMPONENT_SWIZZLE_IDENTITY,
      .components.a = VK_COMPONENT_SWIZZLE_IDENTITY,
      .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .subresourceRange.baseMipLevel = 0,
      .subresourceRange.levelCount = 1,
      .subresourceRange.baseArrayLayer = 0,
      .subresourceRange.layerCount = 1,
    };
    
    if (vkCreateImageView(vulkan_context->logical_device,
                          &image_view_create_info,
                          NULL,
                          &(vulkan_context->swap_chain_image_views[i])
                          ) != VK_SUCCESS)
    {
      printf("could not create swap_chain\n");
      assert(0);
    }
    
  }
  
  // Creating a Graphics pipline
  // Shader stuff
  // TODO(shmoich): Some code duplication is happening here abstract it out
  LoadedFile vert_shader_file = DEBUG_Win32_ReadEntireFile("shaders/vert.spv");
  VkShaderModuleCreateInfo vert_shader_info = {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = vert_shader_file.size,
    .pCode = (uint32_t*)vert_shader_file.buffer
  };
  VkShaderModule vert_shader_module;
  if (vkCreateShaderModule(vulkan_context->logical_device, &vert_shader_info, NULL, &vert_shader_module) != VK_SUCCESS)
  {
    printf("Could not create vert shader module\n");
    assert(0);
  }
  
  LoadedFile frag_shader_file = DEBUG_Win32_ReadEntireFile("shaders/frag.spv");
  VkShaderModuleCreateInfo frag_shader_info = {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = frag_shader_file.size,
    .pCode = (uint32_t*)frag_shader_file.buffer
  };
  VkShaderModule frag_shader_module;
  if (vkCreateShaderModule(vulkan_context->logical_device, &frag_shader_info, NULL, &frag_shader_module) != VK_SUCCESS)
  {
    printf("Could not create frag shader module\n");
    assert(0);
  }
  
  // glue the shaders together into a pipline
  
  VkPipelineShaderStageCreateInfo vert_shader_stage_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage = VK_SHADER_STAGE_VERTEX_BIT,
    .module = vert_shader_module,
    .pName = "main"
  };
  
  VkPipelineShaderStageCreateInfo frag_shader_stage_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
    .module = frag_shader_module,
    .pName = "main"
  };
  
  VkPipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_info, frag_shader_stage_info};
  
  // Fixed functions
  
  
  
  // Vertex input
  VkVertexInputAttributeDescription* vertex_attribute_descriptions = get_attribute_descriptions();
  VkVertexInputBindingDescription vertex_binding_description = get_binding_description();
  
  // Vertex Attributes, etc would be set up here
  VkPipelineVertexInputStateCreateInfo vertex_input_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    .vertexBindingDescriptionCount = 1,
    .pVertexBindingDescriptions = &vertex_binding_description,
    .vertexAttributeDescriptionCount = 2,
    .pVertexAttributeDescriptions = vertex_attribute_descriptions
  };
  
  // do we want to draw strips, tringles, lines?, etc
  VkPipelineInputAssemblyStateCreateInfo input_assembly_info= {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    .primitiveRestartEnable = VK_FALSE
  };
  
  // Viewport
  // I'll probably want to abstract this
  VkViewport viewport = {
    .x = 0.0f,
    .y = 0.0f,
    .width = (float) vulkan_context->swap_chain_extent.width,
    .height = (float) vulkan_context->swap_chain_extent.height,
    .minDepth = 0.0f,
    .maxDepth = 1.0f
  };
  
  VkRect2D scissor = 
  {
    .offset = {0, 0},
    .extent = vulkan_context->swap_chain_extent
  };
  
  VkPipelineViewportStateCreateInfo viewport_state_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
    .viewportCount = 1,
    .pViewports = &viewport,
    .scissorCount = 1,
    .pScissors = &scissor
  };
  
  VkPipelineRasterizationStateCreateInfo rasterizer_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
    .depthClampEnable = VK_FALSE,
    .rasterizerDiscardEnable = VK_FALSE,
    .polygonMode = VK_POLYGON_MODE_FILL,
    .lineWidth = 1.0f,
    .cullMode = VK_CULL_MODE_BACK_BIT,
    .frontFace = VK_FRONT_FACE_CLOCKWISE,
    .depthBiasEnable = VK_FALSE, // TODO(shmoich
 ): figure out what this is in relation to the depth buffer
    .depthBiasConstantFactor = 0.0f, // Optional
    .depthBiasClamp = 0.0f, // Optional
    .depthBiasSlopeFactor = 0.0f // Optional
  };
  
  VkPipelineMultisampleStateCreateInfo multisampling_info = 
  {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
    .sampleShadingEnable = VK_FALSE,
    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    .minSampleShading = 1.0f, // Optional
    .pSampleMask = NULL,
    .alphaToCoverageEnable = VK_FALSE,
    .alphaToOneEnable = VK_FALSE
  };
  
  // TODO(shmoich): depth and stencil testing
  
  // COLOR BLENDING
  // TODO(shmoic): learn about color space better
  // Attached per frame buffer
  VkPipelineColorBlendAttachmentState color_blend_attachment_info =
  {
    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    .blendEnable = VK_FALSE,
    .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
    .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
    .colorBlendOp = VK_BLEND_OP_ADD,
    
    .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
    .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
    .alphaBlendOp = VK_BLEND_OP_ADD,
    
    .blendEnable = VK_TRUE,
    .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
    .colorBlendOp = VK_BLEND_OP_ADD,
    .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
    .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
    .alphaBlendOp = VK_BLEND_OP_ADD
  };
  
  VkPipelineColorBlendStateCreateInfo color_blend_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
    .logicOpEnable = VK_FALSE, // bit wise blending
    .logicOp = VK_LOGIC_OP_COPY,
    .attachmentCount = 1,
    .pAttachments = &color_blend_attachment_info,
    .blendConstants[0] = 0.0f,
    .blendConstants[1] = 0.0f,
    .blendConstants[2] = 0.0f,
    .blendConstants[3] = 0.0f
  };
  
  // NOTE(shmoich): parts of the pipeline that can be changed (with out building the pipeline)
  VkDynamicState dynamic_states[] = {
    VK_DYNAMIC_STATE_VIEWPORT, // we can change the viewport
    VK_DYNAMIC_STATE_LINE_WIDTH // we can change line width
  };
  
  VkPipelineDynamicStateCreateInfo dynamic_state_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
    .dynamicStateCount = 2,
    .pDynamicStates = dynamic_states
  };
  
  // USED for shader uniforms
  VkPipelineLayoutCreateInfo pipeline_layout_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 0,
    .pSetLayouts = NULL,
    .pushConstantRangeCount = 0,
    .pPushConstantRanges = NULL
  };
  
  if (vkCreatePipelineLayout(vulkan_context->logical_device,
                             &pipeline_layout_info,
                             VK_NULL_ALLOC,
                             &(vulkan_context->pipeline_layout)) != VK_SUCCESS)
  {
    printf("could not make pipeline layouts \n");
    assert(0);
  }
  
  // render pass
  VkAttachmentDescription color_attachment = {
    .format = vulkan_context->swap_chain_format,
    .samples = VK_SAMPLE_COUNT_1_BIT, //No Multisampling
    // what to do before and after rendering
    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, // Clear screen
    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    // Not using the stencil buffer right now so we "dont care"
    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
  };
  
  // subpasses used for post processing
  VkAttachmentReference color_attachment_ref = 
  {
    .attachment = 0, //attachment index is refferenced byt the output of the frag shader location(layout=0)
    .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
  };
  
  VkSubpassDescription subpass = {
    .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
    .colorAttachmentCount = 1,
    .pColorAttachments = &color_attachment_ref
  };
  
  VkSubpassDependency dependency =
  {
    .srcSubpass = VK_SUBPASS_EXTERNAL,
    .dstSubpass = 0,
    
    .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    .srcAccessMask = 0,
    
    .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
  };
  
  VkRenderPassCreateInfo render_pass_info =
  {
    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
    .attachmentCount = 1,
    .pAttachments = &color_attachment,
    .subpassCount = 1,
    .pSubpasses = &subpass,
    .dependencyCount = 1,
    .pDependencies = &dependency
  };
  
  if (vkCreateRenderPass(vulkan_context->logical_device, &render_pass_info, NULL, &(vulkan_context->render_pass)) != VK_SUCCESS) {
    printf("could not make renderpass \n");
    assert(0);
  }
  
  VkGraphicsPipelineCreateInfo graphics_pipeline_info = {
    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
    .stageCount = 2,
    .pStages = shader_stages,
    .pVertexInputState = &vertex_input_info,
    .pInputAssemblyState = &input_assembly_info,
    .pViewportState = &viewport_state_info,
    .pRasterizationState = &rasterizer_info,
    .pMultisampleState = &multisampling_info,
    .pDepthStencilState = NULL, // Optional
    .pColorBlendState = &color_blend_info,
    .pDynamicState = NULL, // Optional
    .layout = vulkan_context->pipeline_layout,
    .renderPass = vulkan_context->render_pass,
    .subpass = 0,
    .basePipelineHandle = VK_NULL_HANDLE, // make new pipeline from existing one
    .basePipelineIndex = -1
  };
  
  if(vkCreateGraphicsPipelines(vulkan_context->logical_device,
                               VK_NULL_HANDLE,
                               1,
                               &graphics_pipeline_info,
                               VK_NULL_ALLOC,
                               &(vulkan_context->graphics_pipeline)) != VK_SUCCESS)
  {
    printf("could not create graphics pipeline\n");
    assert(0);
  }
  
  // frambuffers
  vulkan_context->frame_buffers = ScratchAlloc(&permanent_storage, sizeof(VkFramebuffer) * vulkan_context->frame_buffer_count);
  for (size_t i = 0; i < vulkan_context->frame_buffer_count; i++)
  {
    VkImageView attachments[] = {
      vulkan_context->swap_chain_image_views[i]
    };
    
    VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = vulkan_context->render_pass,
      .attachmentCount = 1,
      .pAttachments = attachments,
      .width = vulkan_context->swap_chain_extent.width,
      .height = vulkan_context->swap_chain_extent.height,
      .layers = 1
    };
    
    if (vkCreateFramebuffer(vulkan_context->logical_device,
                            &framebuffer_info,
                            VK_NULL_ALLOC,
                            &(vulkan_context->frame_buffers[i])) != VK_SUCCESS)
    {
      printf("could not create frame_buffers\n");
      assert(0);
    }
  }
  
  // Command Buffers
  VkCommandPoolCreateInfo pool_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .queueFamilyIndex = vulkan_context->graphics_queue_family_index,
    .flags = 0 // Optional
  };
  if (vkCreateCommandPool(vulkan_context->logical_device, &pool_info, NULL, &vulkan_context->command_pool) != VK_SUCCESS)
  {
    printf("could not create command pool\n");
    assert(0);
  }
  
  uint32_t command_buffer_count = vulkan_context->frame_buffer_count;
  vulkan_context->command_buffers = ScratchAlloc(&permanent_storage, sizeof(VkFramebuffer) * vulkan_context->frame_buffer_count);
  
  VkCommandBufferAllocateInfo alloc_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = vulkan_context->command_pool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = command_buffer_count
  };
  
  if (vkAllocateCommandBuffers(vulkan_context->logical_device, &alloc_info, vulkan_context->command_buffers) != VK_SUCCESS) {
    printf("could not create command pool\n");
    assert(0);
  }
  
  // prep command buffers?
  for (size_t i = 0; i < command_buffer_count; i++) 
  {
    VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
      .pInheritanceInfo = NULL
    };
    
    if (vkBeginCommandBuffer(vulkan_context->command_buffers[i], &begin_info) != VK_SUCCESS) 
    {
      printf("could not create command buffers\n");
      assert(0);
    }
    
    VkClearValue clear_color = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    VkRenderPassBeginInfo render_pass_info = 
    {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = vulkan_context->render_pass,
      .framebuffer = vulkan_context->frame_buffers[i],
      .renderArea.offset = {0, 0},
      .renderArea.extent = vulkan_context->swap_chain_extent,
      .clearValueCount = 1,
      .pClearValues = &clear_color 
    };
    
    vkCmdBeginRenderPass(vulkan_context->command_buffers[i], &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(vulkan_context->command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, vulkan_context->graphics_pipeline);
    vkCmdDraw(vulkan_context->command_buffers[i], 3, 1, 0, 0);
    
    vkCmdEndRenderPass(vulkan_context->command_buffers[i]);
    if (vkEndCommandBuffer(vulkan_context->command_buffers[i]) != VK_SUCCESS) {
      printf("failed to record command buffer!");
      assert(0);
    }
  }
  
  vkDestroyShaderModule(vulkan_context->logical_device, vert_shader_module, NULL);
  DEBUG_Win32_FreeFileMemory(vert_shader_file.buffer);
  vkDestroyShaderModule(vulkan_context->logical_device, frag_shader_module, NULL);
  DEBUG_Win32_FreeFileMemory(frag_shader_file.buffer);
  
  ScratchFreeAll(&scratch);
  ScratchDestroy(&scratch);
}

void DestroySwapchain(VulkanContext* vulkan_context)
{
  // destroy framebuffers
  for (size_t i = 0 ; i < vulkan_context->frame_buffer_count; i++)
  {
    vkDestroyFramebuffer(vulkan_context->logical_device, vulkan_context->frame_buffers[i], VK_NULL_ALLOC);
  }
  // destroy pipline
  vkDestroyPipeline(vulkan_context->logical_device, vulkan_context->graphics_pipeline, VK_NULL_ALLOC);
  // destroy pipline layout
  vkDestroyPipelineLayout(vulkan_context->logical_device, vulkan_context->pipeline_layout, VK_NULL_ALLOC);
  // destroy renderpass
  vkDestroyRenderPass(vulkan_context->logical_device, vulkan_context->render_pass, VK_NULL_ALLOC);
  // clean up images
  for (size_t i = 0 ; i < vulkan_context->frame_buffer_count; i++)
  {
    vkDestroyImageView(vulkan_context->logical_device, vulkan_context->swap_chain_image_views[i], VK_NULL_ALLOC);
  }
  vkDestroySwapchainKHR(vulkan_context->logical_device, vulkan_context->swap_chain, NULL);
}

void RecreateSwapchain(VulkanContext* vulkan_context)
{
  vkDeviceWaitIdle(vulkan_context->logical_device);
  DestroySwapchain(vulkan_context);
  CreateSwapchain(vulkan_context);
}


void Win32_InitVulkanContext(VulkanContext* vulkan_context, HWND window, HINSTANCE win32_instance, char* program_name)
{
  Scratch scratch = ScratchInit(1024 * 100);
  
  // Instance Creation
  vulkan_context->instance = VK_NULL_HANDLE;
  VkApplicationInfo app_info;
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pNext = NULL;
  app_info.pApplicationName = program_name;
  app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  app_info.pEngineName = program_name;
  app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
  app_info.apiVersion = VK_API_VERSION_1_2;
  
  // instance extensions
  uint32_t instance_extension_count;
  vkEnumerateInstanceExtensionProperties(NULL, &instance_extension_count, NULL);
  VkExtensionProperties* instance_extension_props = (VkExtensionProperties*) ScratchAlloc(&scratch, (sizeof(VkExtensionProperties) * instance_extension_count));
  vkEnumerateInstanceExtensionProperties(NULL, &instance_extension_count, instance_extension_props);
  
  // if we have validation layers
  instance_extension_count += 1;
  char** instance_extension_names = ScratchAlloc(&scratch, sizeof(char*) * (instance_extension_count));
  for(size_t i=0; i < instance_extension_count - 1; i++)
  {
    instance_extension_names[i] = strdup(instance_extension_props[i].extensionName);
  }
  instance_extension_names[instance_extension_count - 1] = strdup(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  
  // validation layers
  uint32_t validation_layers_count;
  vkEnumerateInstanceLayerProperties(&validation_layers_count, 0);
  VkLayerProperties* validation_layers_props = ScratchAlloc(&scratch, sizeof(VkLayerProperties) * validation_layers_count);
  vkEnumerateInstanceLayerProperties(&validation_layers_count, validation_layers_props);
  char** validation_layers_names = ScratchAlloc(&scratch, sizeof(char*) * validation_layers_count);
  for(size_t i=0; i < validation_layers_count; i++)
  {
    validation_layers_names[i] = strdup(validation_layers_props[i].layerName);
  }
  
  VkDebugUtilsMessengerCreateInfoEXT debug_messenger_info = {
    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
    .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
    .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
    .pfnUserCallback = DEBUG_Vulkan_ValidationLayerCallBack,
    .pUserData = NULL, // Optional
  };
  
  VkInstanceCreateInfo instance_create_info = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pNext = &debug_messenger_info,
    .flags = 0,
    .pApplicationInfo = &app_info,
    .enabledExtensionCount = instance_extension_count,
    .ppEnabledExtensionNames = instance_extension_names,
    .enabledLayerCount = validation_layers_count,
    .ppEnabledLayerNames = validation_layers_names
  };
  if(vkCreateInstance(&instance_create_info, NULL, &(vulkan_context->instance)) != VK_SUCCESS) 
  {
    printf("could not create vulkan instance: \n");
    exit(1);
  }
  
  
  
#if 0
  PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(vulkan_context->instance, "vkCreateDebugUtilsMessengerEXT");
  if(vkCreateDebugUtilsMessengerEXT(vulkan_context->instance,
                                    &debug_messenger_info,
                                    VK_NULL_ALLOC,
                                    &DEBUG_Vulkan_ValidationLayerCallBack) != VK_SUCCESS)
  {
    printf("could not create debug callback \n");
    exit(1);
  }
#endif
  
  // Window surface creation
  // TODO(shmoich): I think this is  the only windows specific code this function?
  vulkan_context->surface = VK_NULL_HANDLE;
  VkWin32SurfaceCreateInfoKHR win32_surface_info;
  win32_surface_info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
  win32_surface_info.pNext = NULL;
  win32_surface_info.flags = 0;
  win32_surface_info.hinstance = win32_instance; // NOTE(BB): w32 instance NOT vulkan instance
  win32_surface_info.hwnd = window;
  if(vkCreateWin32SurfaceKHR(vulkan_context->instance, &win32_surface_info,
                             NULL,
                             &vulkan_context->surface))
  {
    printf("window surface creation failed!\n");
    assert(0);
  }
  
  
  // Physical Device Creation
  vulkan_context->physical_device = VK_NULL_HANDLE;
  uint32_t vulkan_physical_device_count = 0;
  vkEnumeratePhysicalDevices(vulkan_context->instance, &vulkan_physical_device_count, NULL);
  VkPhysicalDevice* vulkan_physical_devices = ScratchAlloc(&scratch, sizeof(VkPhysicalDevice) * vulkan_physical_device_count);
  vkEnumeratePhysicalDevices(vulkan_context->instance, &vulkan_physical_device_count, vulkan_physical_devices);
  for(size_t i=0; i<vulkan_physical_device_count; i++) 
  {
    VkPhysicalDeviceProperties device_check_props;
    VkPhysicalDeviceFeatures device_check_features;
    vkGetPhysicalDeviceProperties(vulkan_physical_devices[i], &device_check_props);
    vkGetPhysicalDeviceFeatures(vulkan_physical_devices[i], &device_check_features);
    if (device_check_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && device_check_features.geometryShader) 
    {
      vulkan_context->physical_device = vulkan_physical_devices[i];
    }
  }
  if(vulkan_context->physical_device == VK_NULL_HANDLE)
  {
    printf("no device available\n");
    assert(0);
  }
  
  // Device extenssion checks
  uint32_t required_device_extension_count = 1;
  char* required_device_extenstions[1] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
  bool required_function_support = true;
  uint32_t device_extension_count;
  vkEnumerateDeviceExtensionProperties(vulkan_context->physical_device, NULL, &device_extension_count, NULL);
  VkExtensionProperties* device_extension_props = ScratchAlloc(&scratch, sizeof(VkExtensionProperties) * device_extension_count);
  vkEnumerateDeviceExtensionProperties(vulkan_context->physical_device, NULL, &device_extension_count, device_extension_props);
  for(size_t i = 0; i < required_device_extension_count; i++)
  {
    bool has_extension = false;
    
    for(size_t j = 0; j < device_extension_count; j++)
    {
      if (strcmp(required_device_extenstions[i] , device_extension_props[j].extensionName) == 0)
      {
        has_extension = true;
        break;
      }
    }
    if(!has_extension)
    {
      required_function_support = false;
      break;
    }
  }
  
  if (required_function_support == false)
  {
    printf("could not find swap chain extension for chosen physical device\n");
    assert(0);
  }
  
  //////////////////
  // Queue families (ie the types of queues)
  // were just  consturnd with the Graphics bit for a trianglle right now
  // but there seems to be bits for things like compute aswell
  // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkQueueFlagBits.html
  // "magic number" for now will define a macro for clarity later
  // TODO(shmoich): do we need these after queue creation?
  
  vulkan_context->present_queue_family_index = VK_MAX_QUEUE_FAMILY_INDEX;
  vulkan_context->graphics_queue_family_index = VK_MAX_QUEUE_FAMILY_INDEX; 
  
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(vulkan_context->physical_device, &queue_family_count, NULL);
  VkQueueFamilyProperties* queue_family_props = ScratchAlloc(&scratch, sizeof(VkQueueFamilyProperties) * queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(vulkan_context->physical_device, &queue_family_count, queue_family_props);
  
  VkBool32 present_support = false;
  
  for(size_t i=0; i < queue_family_count; i++)
  {
    vkGetPhysicalDeviceSurfaceSupportKHR(vulkan_context->physical_device, i, vulkan_context->surface, &present_support);
    
    if(queue_family_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
    {
      vulkan_context->graphics_queue_family_index = i;
    }
    
    if(present_support)
    {
      vulkan_context->present_queue_family_index = i;
    }
    
    if(vulkan_context->graphics_queue_family_index != VK_MAX_QUEUE_FAMILY_INDEX && vulkan_context->present_queue_family_index != VK_MAX_QUEUE_FAMILY_INDEX )
      break;
  }
  
  if(vulkan_context->graphics_queue_family_index == VK_MAX_QUEUE_FAMILY_INDEX)
  {
    printf("no grapics family available\n");
    assert(0);
  }
  
  if(vulkan_context->present_queue_family_index == VK_MAX_QUEUE_FAMILY_INDEX)
  {
    printf("no grapics family available\n");
    assert(0);
  }
  
  // Logical Device Creation
  // Disable all settings for now
  VkPhysicalDeviceFeatures device_features;
  device_features.robustBufferAccess = VK_FALSE;
  device_features.fullDrawIndexUint32 = VK_FALSE;
  device_features.imageCubeArray = VK_FALSE;
  device_features.independentBlend = VK_FALSE;
  device_features.geometryShader = VK_FALSE;
  device_features.tessellationShader = VK_FALSE;
  device_features.sampleRateShading = VK_FALSE;
  device_features.dualSrcBlend = VK_FALSE;
  device_features.logicOp = VK_FALSE;
  device_features.multiDrawIndirect = VK_FALSE;
  device_features.drawIndirectFirstInstance = VK_FALSE;
  device_features.depthClamp = VK_FALSE;
  device_features.depthBiasClamp = VK_FALSE;
  device_features.fillModeNonSolid = VK_FALSE;
  device_features.depthBounds = VK_FALSE;
  device_features.wideLines = VK_FALSE;
  device_features.largePoints = VK_FALSE;
  device_features.alphaToOne = VK_FALSE;
  device_features.multiViewport = VK_FALSE;
  device_features.samplerAnisotropy = VK_FALSE;
  device_features.textureCompressionETC2 = VK_FALSE;
  device_features.textureCompressionASTC_LDR = VK_FALSE;
  device_features.textureCompressionBC = VK_FALSE;
  device_features.occlusionQueryPrecise = VK_FALSE;
  device_features.pipelineStatisticsQuery = VK_FALSE;
  device_features.vertexPipelineStoresAndAtomics = VK_FALSE;
  device_features.fragmentStoresAndAtomics = VK_FALSE;
  device_features.shaderTessellationAndGeometryPointSize = VK_FALSE;
  device_features.shaderImageGatherExtended = VK_FALSE;
  device_features.shaderStorageImageExtendedFormats = VK_FALSE;
  device_features.shaderStorageImageMultisample = VK_FALSE;
  device_features.shaderStorageImageReadWithoutFormat = VK_FALSE;
  device_features.shaderStorageImageWriteWithoutFormat = VK_FALSE;
  device_features.shaderUniformBufferArrayDynamicIndexing = VK_FALSE;
  device_features.shaderSampledImageArrayDynamicIndexing = VK_FALSE;
  device_features.shaderStorageBufferArrayDynamicIndexing = VK_FALSE;
  device_features.shaderStorageImageArrayDynamicIndexing = VK_FALSE;
  device_features.shaderClipDistance = VK_FALSE;
  device_features.shaderCullDistance = VK_FALSE;
  device_features.shaderFloat64 = VK_FALSE;
  device_features.shaderInt64 = VK_FALSE;
  device_features.shaderInt16 = VK_FALSE;
  device_features.shaderResourceResidency = VK_FALSE;
  device_features.shaderResourceMinLod = VK_FALSE;
  device_features.sparseBinding = VK_FALSE;
  device_features.sparseResidencyBuffer = VK_FALSE;
  device_features.sparseResidencyImage2D = VK_FALSE;
  device_features.sparseResidencyImage3D = VK_FALSE;
  device_features.sparseResidency2Samples = VK_FALSE;
  device_features.sparseResidency4Samples = VK_FALSE;
  device_features.sparseResidency8Samples = VK_FALSE;
  device_features.sparseResidency16Samples = VK_FALSE;
  device_features.sparseResidencyAliased = VK_FALSE;
  device_features.variableMultisampleRate = VK_FALSE;
  device_features.inheritedQueries  = VK_FALSE;
  
  vulkan_context->logical_device = VK_NULL_HANDLE;
  
  VkDeviceQueueCreateInfo graphics_queue_create_info;
  graphics_queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  graphics_queue_create_info.pNext = NULL;
  graphics_queue_create_info.flags = 0;
  graphics_queue_create_info.queueFamilyIndex = vulkan_context->graphics_queue_family_index;
  graphics_queue_create_info.queueCount = 1;
  float graphics_queue_priority =  1.0f;
  graphics_queue_create_info.pQueuePriorities = &graphics_queue_priority;
  
  VkDeviceQueueCreateInfo present_queue_create_info;
  present_queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  present_queue_create_info.pNext = NULL;
  present_queue_create_info.flags = 0;
  present_queue_create_info.queueFamilyIndex = vulkan_context->present_queue_family_index;
  present_queue_create_info.queueCount = 1;
  float present_queue_priority =  1.0f;
  present_queue_create_info.pQueuePriorities = &present_queue_priority;
  
  size_t vulkan_queue_info_count = 2;
  VkDeviceQueueCreateInfo *vulkan_queue_infos = ScratchAlloc(&scratch, sizeof(VkDeviceQueueCreateInfo) * vulkan_queue_info_count);
  vulkan_queue_infos[0] = graphics_queue_create_info;
  vulkan_queue_infos[1] = present_queue_create_info;
  
  VkDeviceCreateInfo logical_device_create_info;
  logical_device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  logical_device_create_info.pNext = NULL;
  logical_device_create_info.flags = 0;
  logical_device_create_info.queueCreateInfoCount = 1;
  logical_device_create_info.pQueueCreateInfos = vulkan_queue_infos;
  // validation
  logical_device_create_info.enabledLayerCount = validation_layers_count;
  logical_device_create_info.ppEnabledLayerNames = validation_layers_names;
  
  logical_device_create_info.enabledExtensionCount = required_device_extension_count;
  logical_device_create_info.ppEnabledExtensionNames = required_device_extenstions;
  logical_device_create_info.pEnabledFeatures = &device_features;
  
  if(vkCreateDevice(vulkan_context->physical_device, &logical_device_create_info, NULL, &vulkan_context->logical_device) != VK_SUCCESS)
  {
    printf("could not create logical device\n");
    assert(0);
  }
  
  vulkan_context->graphics_queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(vulkan_context->logical_device, vulkan_context->graphics_queue_family_index, 0, &(vulkan_context->graphics_queue));
  if(vulkan_context->graphics_queue == VK_NULL_HANDLE)
  {
    printf("could not create vulkan graphics queue\n");
    assert(0);
  }
  
  vulkan_context->present_queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(vulkan_context->logical_device, vulkan_context->present_queue_family_index, 0, &(vulkan_context->present_queue));
  if(vulkan_context->present_queue == VK_NULL_HANDLE)
  {
    printf("could not create vulkan present queue\n");
    assert(0);
  }
  
  CreateSwapchain(vulkan_context);
  
  
  VkSemaphoreCreateInfo semaphore_info = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
  };
  VkFenceCreateInfo fence_info = {
    .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    .flags = VK_FENCE_CREATE_SIGNALED_BIT
  };
  vulkan_context->image_available_semaphores = ScratchAlloc(&permanent_storage, sizeof(VkSemaphore) * VK_MAX_FRAMES_IN_FLIGHT);
  vulkan_context->render_finished_semaphores = ScratchAlloc(&permanent_storage, sizeof(VkSemaphore) * VK_MAX_FRAMES_IN_FLIGHT);
  vulkan_context->in_flight_fences = ScratchAlloc(&permanent_storage, sizeof(VkFence) * VK_MAX_FRAMES_IN_FLIGHT);
  vulkan_context->images_in_flight = ScratchAlloc(&permanent_storage, sizeof(VkFence) * vulkan_context->frame_buffer_count); 
  
  for (size_t i = 0; i < VK_MAX_FRAMES_IN_FLIGHT; i++)
  {
    if (
        vkCreateSemaphore(vulkan_context->logical_device,
                          &semaphore_info,
                          NULL,
                          &(vulkan_context->image_available_semaphores[i]))
        ||
        vkCreateSemaphore(vulkan_context->logical_device,
                          &semaphore_info,
                          NULL,
                          &(vulkan_context->render_finished_semaphores[i]))
        ||
        vkCreateFence(vulkan_context->logical_device,
                      &fence_info,
                      NULL,
                      &(vulkan_context->in_flight_fences[i]))
        != VK_SUCCESS
        ) 
    {
      printf("failed to create semaphores!");
      assert(0);
    }
  }
  
  vulkan_context->current_frame = 0;
  
  ScratchFreeAll(&scratch);
  ScratchDestroy(&scratch);
}


void Win32Vulkan_DrawFrame(VulkanContext* vulkan_context)
{
  vkWaitForFences(vulkan_context->logical_device, 1 , &(vulkan_context->in_flight_fences[vulkan_context->current_frame]), VK_TRUE, UINT64_MAX);
  
  
  uint32_t image_index;
  VkResult result =vkAcquireNextImageKHR(vulkan_context->logical_device,
                                         vulkan_context->swap_chain,
                                         UINT64_MAX,
                                         vulkan_context->image_available_semaphores[vulkan_context->current_frame],
                                         NULL,
                                         &image_index);
  
  
  if(result == VK_ERROR_OUT_OF_DATE_KHR)
  {
    RecreateSwapchain(vulkan_context);
    return;
  }
  
  
  if (vulkan_context->images_in_flight[image_index] != VK_NULL_HANDLE)
  {
    vkWaitForFences(vulkan_context->logical_device,
                    1,
                    &(vulkan_context->images_in_flight[image_index]),
                    VK_TRUE, UINT64_MAX);
  }
  
  
  VkSemaphore wait_semaphores[] = {vulkan_context->image_available_semaphores[vulkan_context->current_frame]};
  VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  VkSemaphore signal_semaphores[] = {vulkan_context->render_finished_semaphores[vulkan_context->current_frame]};
  
  VkSubmitInfo submit_info = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = wait_semaphores,
    .pWaitDstStageMask = wait_stages,
    
    .commandBufferCount = 1,
    .pCommandBuffers = &(vulkan_context->command_buffers[image_index]),
    
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = signal_semaphores,
  };
  
  
  vkResetFences(vulkan_context->logical_device, 1 , &(vulkan_context->in_flight_fences[vulkan_context->current_frame]));
  
  
  if(vkQueueSubmit(vulkan_context->graphics_queue,
                   1, //submitCount
                   &submit_info,
                   vulkan_context->in_flight_fences[vulkan_context->current_frame])
     != VK_SUCCESS)
  {
    printf("failed to submit queue!");
    assert(0);
  }
  
  VkSwapchainKHR swap_chains[] = {vulkan_context->swap_chain};
  
  VkPresentInfoKHR present_info = {
    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = signal_semaphores,
    .swapchainCount = 1,
    .pSwapchains = swap_chains,
    .pImageIndices = &image_index,
    .pResults = NULL,
  };
  
  vkQueuePresentKHR(vulkan_context->present_queue, &present_info);
  vkQueueWaitIdle(vulkan_context->present_queue);
  
  vulkan_context->current_frame = (vulkan_context->current_frame + 1) % VK_MAX_FRAMES_IN_FLIGHT;
}

static void Win32_DestroyVulkanContext(VulkanContext* vulkan_context)
{
  
  DestroySwapchain(vulkan_context);
  // destroy command pool
  vkDestroyCommandPool(vulkan_context->logical_device, vulkan_context->command_pool, VK_NULL_ALLOC);
  
  for (size_t i = 0; i < VK_MAX_FRAMES_IN_FLIGHT; i++)
  {
    // destroy render semphore
    vkDestroySemaphore(vulkan_context->logical_device, vulkan_context->render_finished_semaphores[i], VK_NULL_ALLOC);
    vkDestroySemaphore(vulkan_context->logical_device, vulkan_context->image_available_semaphores[i], VK_NULL_ALLOC);
    
    // other fence
    // TODO(shmoich
 ): hist is it
    vkDestroyFence(vulkan_context->logical_device, vulkan_context->in_flight_fences[i], VK_NULL_ALLOC);
  }
  
  
  
  
  vkDestroyDevice(vulkan_context->logical_device, NULL);
  vkDestroySurfaceKHR(vulkan_context->instance, vulkan_context->surface, NULL);
  
  vkDestroyInstance(vulkan_context->instance, NULL);
}

// global
WINDOWPLACEMENT g_wpPrev = { sizeof(g_wpPrev) };
static void Win32_ToggleFullscreen(HWND window)
{
  DWORD dwStyle = GetWindowLong(window, GWL_STYLE);
  if (dwStyle & WS_OVERLAPPEDWINDOW)
  { 
    MONITORINFO mi = { sizeof(mi) };
    
    if (GetWindowPlacement(window, &g_wpPrev) && GetMonitorInfo(MonitorFromWindow(window, MONITOR_DEFAULTTOPRIMARY), &mi)) 
    {
      SetWindowLong(window,
                    GWL_STYLE,
                    dwStyle & ~WS_OVERLAPPEDWINDOW);
      
      SetWindowPos(window,
                   HWND_TOP,
                   mi.rcMonitor.left,
                   mi.rcMonitor.top,
                   mi.rcMonitor.right - mi.rcMonitor.left,
                   mi.rcMonitor.bottom - mi.rcMonitor.top,
                   SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
    }
  } else {
    SetWindowLong(window, GWL_STYLE,
                  dwStyle | WS_OVERLAPPEDWINDOW);
    SetWindowPlacement(window, &g_wpPrev);
    SetWindowPos(window, NULL, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER |
                 SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
  }
}

static LRESULT CALLBACK Win32_WindowProc(HWND   window,
                                         UINT   message,
                                         WPARAM wParam,
                                         LPARAM lParam)
{
  switch (message)
  {
    case WM_DESTROY:
    {
      global_running = 0;
      PostQuitMessage(0);
      return 0;
    }
    break;
    case WM_SIZE:
    {
      global_window_dims = Win32_GetWindowDims(window);
      return 0;
    }
    break;
    default:
    {
      return DefWindowProc(window, message, wParam, lParam);
    }
  }
  return 0; // default
}

int WINAPI WinMain(HINSTANCE win32_instance,
                   HINSTANCE prev_win32_instance,
                   LPSTR command_line,
                   int command_show)
{
  char* window_class_name =  "Win32_WindowClass";
  
  char* program_name = "Potassium";
  
  WNDCLASSEX window_class;
  window_class.cbSize = sizeof(WNDCLASSEX);
  window_class.style = CS_HREDRAW|CS_VREDRAW;
  window_class.lpfnWndProc = Win32_WindowProc;
  window_class.cbClsExtra = 0;
  window_class.hInstance = win32_instance;
  window_class.hIcon = 0; // should just load defalt icon
  window_class.hCursor = LoadCursor(0, IDC_ARROW);
  window_class.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
  window_class.lpszMenuName = 0;
  window_class.lpszClassName = window_class_name;
  window_class.hIconSm = 0;
  
  
  global_window_dims.width = 400;
  global_window_dims.height = 200;
  
  global_running = true;
  
  
  if(RegisterClassEx(&window_class))
  {
    
    HWND window = CreateWindow(window_class_name,
                               "Potassium",
                               WS_OVERLAPPEDWINDOW,
                               CW_USEDEFAULT, CW_USEDEFAULT,
                               global_window_dims.width, global_window_dims.height,
                               0, 0,
                               win32_instance,
                               0);
    
    
    if(window)
    {
      // TODO(shmoich
   ): this will need to be moved to a stack system at some point for now its fine
      permanent_storage = ScratchInit(1024 * 1024 * 10);
      
      VulkanContext vulkan_context;
      Win32_InitVulkanContext(&vulkan_context, window, win32_instance, program_name);
      
      
      ShowWindow(window, command_show);
      UpdateWindow(window);
      
      // sim render loop
      while(global_running)
      {
        // Input Handling
        MSG message;
        while(PeekMessage(&message, window, 0, 0, PM_REMOVE))
        {
          switch(message.message)
          {
            //case WM_SYSKEYUP:
            //case WM_SYSKEYDOWN:
            case WM_KEYUP:
            //case WM_KEYDOWN:
            {
              switch (message.wParam)
              {
                case VK_ESCAPE:
                {
                  global_running = 0;
                  PostQuitMessage(0);
                }
                break;
                
                case VK_SPACE:
                {
                  Win32_ToggleFullscreen(window);
                }
                break;
              }
            }
            break;
          }
          
          TranslateMessage(&message);
          DispatchMessage(&message);
        }
        // Game logic
        
        // Rendering
        Win32Vulkan_DrawFrame(&vulkan_context);
        
      } // sim render loop end
      
      vkDeviceWaitIdle(vulkan_context.logical_device);
      
      //clean up
      Win32_DestroyVulkanContext(&vulkan_context);
      ScratchDestroy(&permanent_storage);
    }
    else
    {
      printf("Could not create window!\n");
    }
  }
  else
  {
    printf("Could not register window class!\n");
  }
  return 0;
}