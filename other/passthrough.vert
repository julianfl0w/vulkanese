
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 color;
layout (location = 3) in uint index;
layout (location = 4) out vec3 fragColor;

void main() {                         
    gl_Position = vec4(position, 1.0);
    fragColor = color;                
}                                     
