#include "load_data_to_vk.h"

#include <limits>

#include <cstring> // for std::memcpy()
#include <tuple>
#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/to_string.hpp"
namespace lut = labutils;



ModelPack set_up_model(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator, BakedModel const& aModel,
    VkCommandPool& aLoadCmdPool, VkDescriptorPool& aDesPool, VkSampler& aSampler, VkDescriptorSetLayout& descLayout)
{
    ModelPack ret;
    //extract data from BakedModel

    for (auto& mesh : aModel.meshes) {
        std::vector<float> vertexData;

        for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
            vertexData.emplace_back(mesh.positions[i].x);
            vertexData.emplace_back(mesh.positions[i].y);
            vertexData.emplace_back(mesh.positions[i].z);
            vertexData.emplace_back(mesh.texcoords[i].x);
            vertexData.emplace_back(mesh.texcoords[i].y);
            vertexData.emplace_back(mesh.normals[i].x);
            vertexData.emplace_back(mesh.normals[i].y);
            vertexData.emplace_back(mesh.normals[i].z);
        }

        //create buffers
        lut::Buffer vertexGPU = lut::create_buffer(aAllocator, vertexData.size() * sizeof(float),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        lut::Buffer indexGPU = lut::create_buffer(aAllocator, mesh.indices.size() * sizeof(uint32_t),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        lut::Buffer vertexStaging = lut::create_buffer(aAllocator, vertexData.size() * sizeof(float),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        lut::Buffer indexStaging = lut::create_buffer(aAllocator, mesh.indices.size() * sizeof(uint32_t),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

        void* vertPtr = nullptr;
        if (auto const res = vmaMapMemory(aAllocator.allocator, vertexStaging.allocation, &vertPtr); VK_SUCCESS != res)
        {
            throw lut::Error("Mapping memory for writing\n""vmaMapMemory() returned %s", lut::to_string(res).c_str());
        }
        std::memcpy(vertPtr, vertexData.data(), vertexData.size() * sizeof(float));
        vmaUnmapMemory(aAllocator.allocator, vertexStaging.allocation);

        void* indexPtr = nullptr;
        if (auto const res = vmaMapMemory(aAllocator.allocator, indexStaging.allocation, &indexPtr); VK_SUCCESS != res)
        {
            throw lut::Error("Mapping memory for writing\n""vmaMapMemory() returned %s", lut::to_string(res).c_str());
        }
        std::memcpy(indexPtr, mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t));
        vmaUnmapMemory(aAllocator.allocator, indexStaging.allocation);

        lut::Fence uploadComplete = lut::create_fence(aWindow);

        lut::CommandPool uploadPool = lut::create_command_pool(aWindow);
        VkCommandBuffer uploadCmd = lut::alloc_command_buffer(aWindow, uploadPool.handle);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;

        if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
        {
            throw lut::Error("Beginning command buffer recording\n" "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
        }

        VkBufferCopy vcopy{};
        vcopy.size = vertexData.size() * sizeof(float);

        vkCmdCopyBuffer(uploadCmd, vertexStaging.buffer, vertexGPU.buffer, 1, &vcopy);
        lut::buffer_barrier(uploadCmd, vertexGPU.buffer,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

        VkBufferCopy icopy{};
        icopy.size = mesh.indices.size() * sizeof(uint32_t);

        vkCmdCopyBuffer(uploadCmd, indexStaging.buffer, indexGPU.buffer, 1, &icopy);

        lut::buffer_barrier(uploadCmd, indexGPU.buffer,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

        if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
        {
            throw lut::Error("Ennding command buffer recording\n""vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
        }

        // Submit transfer commands
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &uploadCmd;

        if (auto const res = vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
        {
            throw lut::Error("Submitting commands\n" "vkQueueSubmit() returned %s", lut::to_string(res));
        }

        // Wait for commands to finish before we destroy the temporary resources
        // required for the transfers (staging buffers, command pool, ...)
        //
        // The code doesn¡¯t destory the resources implicitly ¨C the resources are
        // destroyed by the destructors of the labutils wrappers for the various
        // objects once we leave the function¡¯s scope.
        if (auto const res = vkWaitForFences(aWindow.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
        {
            throw lut::Error("Waiting for upload to complete\n" "vkWaitForFences() returned %s", lut::to_string(res).c_str());
        }

        Mesh meshData;
        meshData.vertices = std::move(vertexGPU);
        meshData.indexCount = static_cast<uint32_t>(mesh.indices.size());
        meshData.indices = std::move(indexGPU);
        meshData.matID = mesh.materialId;

        ret.meshes.emplace_back(std::move(meshData));
    }

    for (auto& texture : aModel.textures)
    {
        Texture texData;

        lut::Image image = lut::load_image_texture2d(texture.path.c_str(), aWindow, aLoadCmdPool, aAllocator);
        lut::ImageView view = lut::create_image_view_texture2d(aWindow, image.image, VK_FORMAT_R8G8B8A8_SRGB);

        texData.image = std::move(image);
        texData.view = std::move(view);

        ret.textures.emplace_back(std::move(texData));
    }

    //create descriptor sets for every material

    std::vector<VkDescriptorSet> matDescs;
    uint32_t materialCount = static_cast<uint32_t>(aModel.materials.size());
    matDescs.resize(materialCount);
    std::vector<VkDescriptorSetLayout> layouts(materialCount, descLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = aDesPool;
    allocInfo.descriptorSetCount = materialCount;
    allocInfo.pSetLayouts = layouts.data();

    if (auto const res = vkAllocateDescriptorSets(aWindow.device, &allocInfo, matDescs.data()); VK_SUCCESS != res)
    {
        throw lut::Error("Allocating descriptor sets\n" "vkAllocateDescriptorSets() returned %s", lut::to_string(res).c_str());
    }



    for (uint32_t i = 0; i < materialCount; ++i)
    {
        TexParameter texRet;
        texRet.baseColor = aModel.materials[i].baseColor;
        texRet.emissiveColor = aModel.materials[i].emissiveColor;
        texRet.metalness = aModel.materials[i].metalness;
        texRet.roughness = aModel.materials[i].roughness;
        ret.texParameters.emplace_back(texRet);


        lut::Buffer texUBO = lut::create_buffer(aAllocator, sizeof(TexParameter), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        VkDescriptorImageInfo imageInfo[3]{};

        for (uint32_t j = 0; j < 3; ++j)
        {
            imageInfo[j].sampler = aSampler;
            imageInfo[j].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        imageInfo[0].imageView = ret.textures[aModel.materials[i].baseColorTextureId].view.handle;
        imageInfo[1].imageView = ret.textures[aModel.materials[i].roughnessTextureId].view.handle;
        imageInfo[2].imageView = ret.textures[aModel.materials[i].metalnessTextureId].view.handle;
        

        VkDescriptorBufferInfo texUBOInfo{};
        texUBOInfo.buffer = texUBO.buffer;
        texUBOInfo.range = VK_WHOLE_SIZE; 


        VkWriteDescriptorSet desc[4]{};
        for (uint32_t j = 0; j < 3; ++j)
        {
            desc[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            desc[j].dstSet = matDescs[i];
            desc[j].dstBinding = j;
            desc[j].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            desc[j].descriptorCount = 1;
            desc[j].pImageInfo = &imageInfo[j];
        }

        desc[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        desc[3].dstSet = matDescs[i];
        desc[3].dstBinding = 3;
        desc[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        desc[3].descriptorCount = 1;
        desc[3].pBufferInfo = &texUBOInfo;

        constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
        vkUpdateDescriptorSets(aWindow.device, numSets, desc, 0, nullptr);

        ret.texUBOs.emplace_back(std::move(texUBO));
    }

    ret.matDecriptors = std::move(matDescs);

    return ret;

}
