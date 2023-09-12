#version 450

layout (location = 0) out vec4 oColor;

layout(set = 0, binding = 0) uniform sampler2D fullscreenTex;
//layout(set = 0, binding = 1) uniform sampler2D bloomTex;
layout(set = 1, binding = 0) uniform sampler2D bloomTex;

layout( location = 0 ) in vec2 v2fUV;
void main()
{
	vec3 intTexture = texture(fullscreenTex,v2fUV).rgb;
	vec3 bloom = texture(bloomTex,v2fUV).rgb;


	vec3 result = intTexture + bloom;
	//tone mapping
	result /= 1 + result;


	oColor = vec4(result, 1.0);


	//oColor = vec4(1.0f);
}
