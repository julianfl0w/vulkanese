
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require


layout (location = 0) out vec4 outColor;
layout (location = 3) in vec3 fragColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
