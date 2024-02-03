# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL import shaders as gl_shaders
import numpy as np
import torch
import config
import torch.nn.functional as F
import torchvision
import math
from tqdm import tqdm

# os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['LIBGL_ALWAYS_SOFTWARE']='1'

# Vertex shader 
vertex_shader_source = """
#version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_texcoord;

out vec2 texcoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(in_position, 1.0);
    texcoord = in_texcoord;
}
"""

# Fragment shader 
fragment_shader_source = """
#version 330 core

in vec2 texcoord;
out vec4 frag_color;

uniform sampler2D texture_sampler;  // Texture sampler uniform

void main() { 
    vec4 tex_color = texture(texture_sampler, texcoord);  // Sample texture at texcoord
    frag_color = vec4(tex_color.rgb * tex_color.a, tex_color.a); 
}
"""

def render_gl(vertices, indices, texcoords, texture, projection, width, height, camera_poses):
    FLOAT_SZ = 4              
    UINT_SZ = 4               

    n = vertices.shape[0]

    #
    # Initialize Geometry Buffers
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)

    vertex_data = np.concatenate( (vertices, texcoords), -1 )
    vertex_data = np.ascontiguousarray( vertex_data.flatten().astype(np.float32) )
                
    glBufferData( GL_ARRAY_BUFFER, FLOAT_SZ * len(vertex_data), vertex_data, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, FLOAT_SZ * (3 + 2), ctypes.c_void_p(FLOAT_SZ * 0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, FLOAT_SZ * (3 + 2), ctypes.c_void_p(FLOAT_SZ * 3))
    glEnableVertexAttribArray(1)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, UINT_SZ * indices.size, indices.flatten().astype(np.uint32), GL_STATIC_DRAW)

    glBindVertexArray(0)

    #
    # Initialize Texture
    texid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texid)
    h, w = texture.shape[:2]
    texture_ = np.ascontiguousarray(texture.flatten())
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_FLOAT, texture_ )
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameterfv( GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, np.ones(4).astype(np.float32))
    
    glBindTexture(GL_TEXTURE_2D, 0)

    #
    # Initialize Frame Buffer
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo)
    cbo, dbo = glGenTextures(2)

    glBindTexture(GL_TEXTURE_2D, cbo)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, [])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    glBindTexture(GL_TEXTURE_2D, dbo)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, [])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    
    glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, cbo, 0)
    glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, dbo, 0)
    
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print("GL_FRAMEBUFFER not complete!")
        exit()

    #
    # Initialize Shaders
    shader_ids = []
    shader_ids.append(gl_shaders.compileShader(vertex_shader_source, GL_VERTEX_SHADER))
    shader_ids.append(gl_shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER))
    
    # vao_temp = glGenVertexArrays(1)
    # glBindVertexArray(vao_temp) # PyOpenGL bug?
    shader_program = gl_shaders.compileProgram(*shader_ids)
    # glBindVertexArray(0)
    
    model = np.identity(4).astype(np.float32)
    
    glUseProgram(shader_program)
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, model.flatten())
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, projection.flatten())
    glUniform1i(glGetUniformLocation(shader_program, "texture_sampler"), texid)

    frames = []
    for i in tqdm(range( camera_poses.shape[0] )):
        view = np.linalg.inv(camera_poses[i, ...].T).astype(np.float32)
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, view.flatten())

        # Set up OpenGL for rendering the semi-transparent MPI tiles
        glViewport(0, 0, width, height)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glBindVertexArray(vao)
        glActiveTexture(GL_TEXTURE0 + texid)
        glBindTexture(GL_TEXTURE_2D, texid)
        
        glDrawElementsInstanced( GL_TRIANGLES, indices.size, GL_UNSIGNED_INT, ctypes.c_void_p(0), 1)
        
        glFlush()

        # Read rendered image
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo)
        color_buf = glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT)
        rendered_image = np.frombuffer(color_buf, dtype=np.float32)
        rendered_image = rendered_image.reshape((height, width, 4))
        
        frames.append( np.ascontiguousarray(np.flip(rendered_image, axis=0)) )
        
    return frames

def intrinsic_to_opengl(K, w, h, zmin=0.1, zmax=1e5):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    P = np.array([2 * fx/w, 0, 2 * (cx/w) - 1, 0,
                  0, 2 * fy/h, 2 * (cy/h) - 1, 0,
                  0, 0, -(zmax+zmin)/(zmax-zmin), -2 * zmax*zmin/(zmax-zmin),
                  0, 0, -1, 0]).reshape(4, 4)
    return P.T
                                
class TMPIRendererGL(torch.nn.Module):
    def __init__(self, ho, wo):
        super(TMPIRendererGL, self).__init__()
        self.ho = ho
        self.wo = wo

        if 'PYOPENGL_PLATFORM' not in os.environ:
            from pyrender.platforms.pyglet_platform import PygletPlatform
            self.platform = PygletPlatform(wo, ho)
            
        elif os.environ['PYOPENGL_PLATFORM'] == 'egl':
            from pyrender.platforms import egl
            device_id = int(os.environ.get('EGL_DEVICE_ID', '0'))
            egl_device = egl.get_device_by_index(device_id)
            self.platform = egl.EGLPlatform(wo, ho, device=egl_device)
        
        elif os.environ['PYOPENGL_PLATFORM'] == 'osmesa':
            from pyrender.platforms.osmesa import OSMesaPlatform
            self.platform = OSMesaPlatform(wo, ho)
        else:
            raise ValueError('Unsupported PyOpenGL platform: {}'.format(
                os.environ['PYOPENGL_PLATFORM']
            ))
        self.platform.init_context()

    
    def forward(self, mpis, mpi_disp, cam_pose, K, sx, sy):

        # Convert the MPI tiles into an RGBA texture atlas
        mpis = mpis.permute(0, 2, 1, 3, 4, 5).contiguous().squeeze(0)
        b, nt, c, h, w = mpis.shape
        mpi_tiles = mpis.view(b, nt, c * h * w).permute(0, 2, 1)
        tile_sz = h
        pad_sz = int(tile_sz * config.padsz2tile_ratio)
        nt_x, nt_y = np.ceil(self.wo / (tile_sz - pad_sz)).astype(np.int32), np.ceil(self.ho / (tile_sz - pad_sz)).astype(np.int32)
        textures = F.fold( mpi_tiles,
                           output_size=(nt_y * h, nt_x * w),
                           kernel_size=w,
                           stride=w)

        texture_atlas = torchvision.utils.make_grid( textures, padding=0, nrow=np.ceil(np.sqrt(config.num_planes)).astype(np.uint32) )
        ta_height, ta_width = texture_atlas.shape[-2:]

        # Generate the geometry of the MPI tiles, along with texture coordinates into the texture atlas
        u0, v0 = torch.meshgrid( torch.arange(0, nt_x * w, w), torch.arange(0, nt_y * h, h), indexing='xy')
        u_px = torch.stack( (u0, u0 + w, u0, u0 + w), -1).view(1, -1).expand(b, -1)
        v_px = torch.stack( (v0, v0, v0 + h, v0 + h), -1).view(1, -1).expand(b, -1)

        tc_ox, tc_oy = torch.meshgrid( torch.arange(0, np.ceil(config.num_planes / np.ceil(np.sqrt(config.num_planes)) )),
                                       torch.arange(0, np.sqrt(config.num_planes)), indexing='xy' )
        tc_ox = (tc_ox * textures.shape[-1]).view(-1, 1)[:b, ...]
        tc_oy = (tc_oy * textures.shape[-2]).view(-1, 1)[:b, ...]
        texcoords = torch.stack( ( (u_px + tc_ox) / ta_width, (v_px + tc_oy) / ta_height), -1)
                                    
        x0, y0 = sx.view(nt_y, nt_x), sy.view(nt_y, nt_x)
        xc = torch.stack( (x0, x0 + w, x0, x0 + w), -1).view(1, -1).expand(b, -1)
        yc = torch.stack( (y0, y0, y0 + h, y0 + h), -1).view(1, -1).expand(b, -1)
        zc = torch.reciprocal(mpi_disp).squeeze().permute(1, 0).repeat_interleave(4, dim=-1) # 4 for the four corners of each tile
        yc = self.ho - yc - 1

        zc = zc * -1
        zc = zc.clamp(min=-1e8)

        xw = (xc - K[0, 0, 2]) * torch.abs(zc) / K[0, 0, 0]
        yw = (yc - K[0, 1, 2]) * torch.abs(zc) / K[0, 1, 1]
        vertices = torch.stack( (xw, yw, zc), -1)
        indices = torch.tensor( [0, 2, 1, 1, 2, 3] ).view(1, -1).expand(nt_x * nt_y, -1) + (torch.arange(0, nt_x * nt_y).view(-1, 1) * 4)

        # Sort the quads by depth for back-to-front rendering.
        # This is important if we want to render the semi-transparent tiles correctly with OpenGL
        face_z = torch.reciprocal(mpi_disp).squeeze().permute(1, 0).flatten() 
        indices = indices.unsqueeze(0).expand(b, -1, -1) + torch.arange(0, b).view(b, 1, 1) * vertices.shape[1]
        _, o = torch.sort(-face_z)
        indices = indices.view(-1, 6)[o, ...]

        self.platform.make_current()
        
        frames = render_gl( vertices.view(-1, 3).cpu().numpy().astype(np.float32),
                            indices.flatten().cpu().numpy().astype(np.uint32),
                            texcoords.view(-1, 2).cpu().numpy().astype(np.float32),
                            texture_atlas.permute(1, 2, 0).cpu().numpy().astype(np.float32),
                            intrinsic_to_opengl(K[0, ...], self.wo, self.ho, zmax=1e5).astype(np.float32),
                            self.wo,
                            self.ho,
                            cam_pose.cpu().numpy() )
        return frames
        
