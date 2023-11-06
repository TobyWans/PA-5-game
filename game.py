import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GLU import *
import numpy as np
import pyrr
import random
# Constants
SCREEN_SIZE = (800, 600)
DISPLAY_CENTER = (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2)
WIDTH, HEIGHT = SCREEN_SIZE
FPS = 60
# Global variables
camera_pos = [-77, 0, -77]
camera_rot = 45
rotate_angle = 0.1
# Helpers
def create_shader(vertex_filepath, fragment_filepath):
    with open(vertex_filepath, 'r') as f:
        vertex_src = f.readlines()
        
    with open(fragment_filepath, 'r') as f:
        fragment_src = f.readlines()
        
    shader = compileProgram(
        compileShader(vertex_src, GL_VERTEX_SHADER),
        compileShader(fragment_src, GL_FRAGMENT_SHADER)
    )
    
    return shader

def loadMesh(filename: str) -> list[float]:
    v, vt, vn = [], [], []
    vertices = []
    
    with open(filename, "r") as file:
        for line in file:
            words = line.split()
            if len(words) == 0:
                continue
            if words[0] == "v":
                v.append(read_vertex_data(words))
            elif words[0] == "vt":
                vt.append(read_texcoord_data(words))
            elif words[0] == "vn":
                vn.append(read_normal_data(words))
            elif words[0] == "f":
                read_face_data(words, v, vt, vn, vertices)
    return vertices
        
    
def read_vertex_data(words):
    return [
        float(words[1]),
        float(words[2]),
        float(words[3])
    ]
    
def read_texcoord_data(words):
    return [
        float(words[1]),
        float(words[2])
    ]
    
def read_normal_data(words):
    return [
        float(words[1]),
        float(words[2]),
        float(words[3])
    ]

def read_face_data(words, v, vt, vn, vertices):
    triangleCount = len(words) - 3
    for i in range(triangleCount):
        make_corner(words[1], v, vt, vn, vertices)
        make_corner(words[2 + i], v, vt, vn, vertices)
        make_corner(words[3 + i], v, vt, vn, vertices)
    
def make_corner(words, v, vt, vn, vertices):
    v_vt_vn = words.split("/")
    
    for i in v[int(v_vt_vn[0]) - 1]:
        vertices.append(i)
    for i in vt[int(v_vt_vn[1]) - 1]:
        vertices.append(i)
    for i in vn[int(v_vt_vn[2]) - 1]:
        vertices.append(i)

# Classes
class Engine:
    def __init__(self):
        self.init_pg()
        self.init_gl()
        self.init_assets()
                
    def init_pg(self):
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                    pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode(SCREEN_SIZE, pg.OPENGL|pg.DOUBLEBUF)
        self.clock = pg.time.Clock()
    
    def init_gl(self):
        glClearColor(0.1, 0.1, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)
        
    def init_assets(self):
        self.player = Entity([0, 0, -4], [0, 0, 0])
        self.skybox = Cube(10)
        self.player_model = Mesh('models/plane.obj')
        self.skybox_mat = Material('textures/SkySkybox.png')
        self.material = Material('textures/white.png')
        self.shader = create_shader('shaders/vertex.txt', 'shaders/fragment.txt')
        
        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, 'imageTexture'), 0)
        
        projection_transform = pyrr.matrix44.create_perspective_projection_matrix(60.0, float(WIDTH)/float(HEIGHT), 0.1, 10.0, dtype=np.float32)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, 'projection'), 1, GL_FALSE, projection_transform)
        
        glUseProgram(self.shader)
        self.model_matrix_location = glGetUniformLocation(self.shader, 'model')
        
    def render(self, delta_time=0):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)
        
        glUniformMatrix4fv(self.model_matrix_location, 1, GL_FALSE, self.player.get_model_transform())
        self.material.use()
        self.player_model.draw()
        
        self.skybox_mat.use()
        self.skybox.draw()
        
        
    def handle_events(self):
        keys = pg.key.get_pressed()
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.destroy()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.destroy()

        keys = pg.key.get_pressed()
        
        if keys[K_ESCAPE]:
            self.destroy()
            
        if keys[K_LEFT]:
            self.player.pos[0] -= 0.05
        if keys[K_RIGHT]:
            self.player.pos[0] += 0.05
        if keys[K_UP]:
            self.player.pos[1] += 0.05
        if keys[K_DOWN]:
            self.player.pos[1] -= 0.05
        else:
            self.player.rot[1] = 0
    
    def run(self):
        last_time = pg.time.get_ticks()
        while True:
            self.handle_events()
            
            self.player.update()
            
            current_time = pg.time.get_ticks()
            delta_time = (current_time - last_time) / 1000.0
            last_time = current_time
            
            self.render(delta_time=delta_time)
            pg.display.flip()
            pg.display.set_caption(f'FPS: {self.clock.get_fps():.2f}')
            self.clock.tick(FPS)
            
    def destroy(self):
        self.player_model.destroy()
        self.material.destroy()
        glDeleteProgram(self.shader)
        pg.quit()
        quit()

class Entity:
    def __init__(self, pos, rot) -> None:
        self.pos = np.array(pos, dtype=np.float32)
        self.rot = np.array(rot, dtype=np.float32)
        
    def update(self):
        pass
            
    def get_model_transform(self):
        model = pyrr.matrix44.create_identity(dtype=np.float32)
        model = pyrr.matrix44.multiply(model, pyrr.matrix44.create_from_axis_rotation([0, 1, 0], np.radians(self.rot[1]), dtype=np.float32))
        return pyrr.matrix44.multiply(model, pyrr.matrix44.create_from_translation(self.pos, dtype=np.float32))

class Cube:
    # Draw the player as a cube
    def __init__(self, size=1) -> None:
        s = size / 2
        # x, y, z, s, t
        vertices = (
            -s, -s, -s, 0, 0,
             s, -s, -s, 1, 0,
             s,  s, -s, 1, 1,

             s,  s, -s, 1, 1,
            -s,  s, -s, 0, 1,
            -s, -s, -s, 0, 0,

            -s, -s,  s, 0, 0,
             s, -s,  s, 1, 0,
             s,  s,  s, 1, 1,

             s,  s,  s, 1, 1,
            -s,  s,  s, 0, 1,
            -s, -s,  s, 0, 0,

            -s,  s,  s, 1, 0,
            -s,  s, -s, 1, 1,
            -s, -s, -s, 0, 1,

            -s, -s, -s, 0, 1,
            -s, -s,  s, 0, 0,
            -s,  s,  s, 1, 0,

             s,  s,  s, 1, 0,
             s,  s, -s, 1, 1,
             s, -s, -s, 0, 1,

             s, -s, -s, 0, 1,
             s, -s,  s, 0, 0,
             s,  s,  s, 1, 0,

            -s, -s, -s, 0, 1,
             s, -s, -s, 1, 1,
             s, -s,  s, 1, 0,

             s, -s,  s, 1, 0,
            -s, -s,  s, 0, 0,
            -s, -s, -s, 0, 1,

            -s,  s, -s, 0, 1,
             s,  s, -s, 1, 1,
             s,  s,  s, 1, 0,

             s,  s,  s, 1, 0,
            -s,  s,  s, 0, 0,
            -s,  s, -s, 0, 1
        )
        self.vertex_count = len(vertices)
        vertices = np.array(vertices, dtype=np.float32)
        
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
        
    def update(self):
        pass
        
    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        
    def destroy(self):
        glDeleteBuffers(1, [self.vbo])
        glDeleteVertexArrays(1, [self.vao])

class Material:
    def __init__(self, filename) -> None:
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        image = pg.image.load(filename).convert()
        width, height = image.get_rect().size
        img_data = pg.image.tostring(image, "RGBA")
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)
        
    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        
    def destroy(self):
        glDeleteTextures(1, [self.tex])

class Mesh:
    def __init__(self, filename) -> None:
        vertices = loadMesh(filename)
        self.vertex_count = len(vertices)
        vertices = np.array(vertices, dtype=np.float32)
        
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))
        
    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        
    def destroy(self):
        glDeleteBuffers(1, [self.vbo])
        glDeleteVertexArrays(1, [self.vao])

if __name__ == "__main__":
    engine = Engine()
    engine.run()
    engine.destroy()