
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#define VERTEX_COUNT 12
#define TRIANGLE_COUNT 4
#define VERTS_PER_TRIANGLE 3
#define SPATIAL_DIMENSIONS 3
#define COLOR_DIMENSIONS 3
// This will be (or has been) replaced by constant definitions
    
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 color;
layout (location = 3) out vec3 fragColor;
// This will be (or has been) replaced by buffer definitions
    
void main() {                         
    gl_Position = vec4(position, 1.0);
    fragColor = color;                
}                                     
