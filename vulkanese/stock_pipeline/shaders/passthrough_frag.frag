
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#define VERTEX_COUNT 12
#define TRIANGLE_COUNT 4
#define VERTS_PER_TRIANGLE 3
#define SPATIAL_DIMENSIONS 3
#define COLOR_DIMENSIONS 3
// This will be (or has been) replaced by constant definitions
    
layout (location = 0) out vec4 outColor;
layout (location = 3) in vec3 fragColor;
// This will be (or has been) replaced by buffer definitions
    


void main() {
    outColor = vec4(fragColor, 1.0);
}
