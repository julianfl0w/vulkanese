#version 450                                             
#extension GL_ARB_separate_shader_objects : enable       
                                                         
layout(location = 3) in vec3 fragColor; 
                                                         
//output always location 0                               
layout(location = 0) out vec4 outColor;                  
                                                         
void main() {                                            
    outColor = vec4(fragColor, 1.0);                     
}                                                        