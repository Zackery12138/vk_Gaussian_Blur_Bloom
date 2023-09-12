#version 450
layout (location = 0) out vec4 oColor;

layout( location = 0 ) in vec2 v2fUV;
layout( set = 0, binding = 0 ) uniform sampler2D brightColorTex;

const int kTaps = 12;
const float PI = 3.14159265358979323846;

float Gaussian(float x, float sigma) {
    return (1.0 / (sqrt(2.0 * PI) * sigma)) * exp(-(x * x) / (2.0 * sigma * sigma));
}

void CalculateGaussianWeights(out float weights[kTaps]) {
    float sum = 0.0;
    for (int i = 0; i < kTaps; ++i) {
        weights[i] = Gaussian(float(i * 2), 9.0); //sigma = 9.0
        sum += (i == 0 ? 1.0 : 2.0) * weights[i];
    }

    // Normalize weights
    for (int i = 0; i < kTaps; ++i) {
        weights[i] /= sum;
    }
}
//linear sampling

void main()
{
    //22 weights for 44*44 footprint
    float weights[kTaps]; 
    CalculateGaussianWeights(weights);

    vec2 tex_offset = 2.0 / textureSize(brightColorTex, 0); // gets size of single texel
    vec3 result = texture(brightColorTex, v2fUV).rgb * (weights[0] + weights[1]);

    for(int i = 2; i < kTaps; i += 2)// 10 times loop
    {
        //linear interpolation
        float offset = (tex_offset.x * i * weights[i] +  tex_offset.x * (i + 1) * weights[i+1]) / (weights[i] + weights[i+1]);  
        result += texture(brightColorTex, v2fUV + vec2(offset , 0.0)).rgb * (weights[i] + weights[i+1]);
        result += texture(brightColorTex, v2fUV - vec2(offset , 0.0)).rgb * (weights[i] + weights[i+1]);
    }

    oColor = vec4(result, 1.0);
}

/*
void main()
{
    float weights[kTaps];
    CalculateGaussianWeights(weights);

    vec2 tex_offset = 2.0 / textureSize(brightColorTex, 0); // gets size of single texel
    vec3 result = texture(brightColorTex, v2fUV).rgb * weights[0];

    for(int i = 1; i < kTaps; ++i)
    {
        result += texture(brightColorTex, v2fUV + vec2(tex_offset.x * i , 0.0)).rgb * weights[i];
        result += texture(brightColorTex, v2fUV - vec2(tex_offset.x * i , 0.0)).rgb * weights[i];
    }

    oColor = vec4(result, 1.0);
}
*/
