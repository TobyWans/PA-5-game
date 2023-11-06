import pygame as pg
from OpenGL.GL import *
import numpy as np
import ctypes
from OpenGL.GL.shaders import compileProgram, compileShader

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

def loadMesh(filepath):
    v , vt, vn = [], [], []
    vertices = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if line.startswith('v '):
                v.append([float(x) for x in line.split()[1:]])
            elif line.startswith('vt '):
                vt.append([float(x) for x in line.split()[1:]])
            elif line.startswith('vn '):
                vn.append([float(x) for x in line.split()[1:]])
            elif line.startswith('f '):
                for face in line.split()[1:]:
                    face = face.split('/')
                    vertex = v[int(face[0])-1] + vt[int(face[1])-1] + vn[int(face[2])-1]
                    vertices.append(vertex)
    return np.array(vertices, dtype=np.float32)

class App:
    def __init__(self) -> None:
        # init python
        self._set_up_pygame()
        # init opengl
        self._set_up_opengl()
        # create assets
        self._create_assets()
        # set one time uniforms
        self._set_onetime_uniforms()
    
    def createShader(self, vertex_filepath, fragment_filepath):
        with open(vertex_filepath, 'r') as f:
            vertex_src = f.readlines()
            
        with open(fragment_filepath, 'r') as f:
            fragment_src = f.readlines()
            
        shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )
        
        return shader
    
    def _set_up_pygame(self):
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode((800, 600), pg.OPENGL | pg.DOUBLEBUF)
        self.clock = pg.time.Clock()
        
    
    def _set_up_opengl(self):
        glClearColor(0.1, 0.2, 0.2, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
    def _create_assets(self):
        self.meshes = {
            "car": Mesh("models/cube.obj"),
        }
        self.materials = {
            "car": Material("textures/texture.png"),
        }
        self.shader = create_shader("shaders/vertex.txt", "shaders/fragment.txt")
        
        self.scene = Scene()
        
    def _set_onetime_uniforms(self):
        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, 'image_texture'), 0)
        
        projection_transform = np.identity(4, dtype=np.float32)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, 'projection'), 1, GL_FALSE, projection_transform)
    
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT)
        
        glUseProgram(self.shader)
        
        view_transform = self.scene.player.get_view_matrix()
        glUniformMatrix4fv(glGetUniformLocation(self.shader, 'view'), 1, GL_FALSE, view_transform)
        
        for entity in self.scene.entities.values():
            for e in entity:
                model_transform = e.get_model_transform()
                glUniformMatrix4fv(glGetUniformLocation(self.shader, 'model'), 1, GL_FALSE, model_transform)
                self.meshes["car"].draw()
        
        pg.display.flip()
    
    def run(self):
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
            glClear(GL_COLOR_BUFFER_BIT)
            
            self.scene.update(self.clock.get_time() / 1000.0)
            
            self.render()
            
            #timing 
            self.clock.tick(60)
        
    def quit(self):
        self.triangle.destroy()
        self.material.destroy()
        glDeleteProgram(self.shader)
        pg.quit()
        exit()
 
class Triangle:
    
    def __init__(self) -> None:
        # x, y, z, r, g, b, s, t
        self.vertices = (
            -0.5, -0.5, 0.0, 1.0 ,0.0, 0.0, 0.0, 1.0,
             0.5, -0.5, 0.0, 0.0 ,1.0, 0.0, 1.0, 1.0,
             0.0,  0.5, 0.0, 0.0 ,0.0, 1.0, 0.5, 0.0
        )
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.vertex_count = 3
        
        self.vao = glGenVertexArrays(1) # vertex array object
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)      # vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))
    
    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
    
    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))

class Entity:
    
    slots = ('position', 'rotation')
    
    def __init__(self, pos: list[float], rot: list[float]) -> None:
        self.pos = np.array(pos, dtype=np.float32)
        self.rot = np.array(rot, dtype=np.float32)
    
    def update(self, delta_time: float):
        pass
    
    def get_model_transform(self):
        model_transform = np.identity(4, dtype=np.float32)
        model_transform = np.matmul(model_transform, self.get_translation_matrix())
        model_transform = np.matmul(model_transform, self.get_rotation_matrix())
        return model_transform

class Car(Entity):
    
    slots = tuple([])
    
    def __init__(self, pos: list[float], rot: list[float]) -> None:
        super().__init__(pos, rot)
    
    def update(self, delta_time: float):
        pass

class PointLight(Entity):
    def __init__(self, pos: list[float], color: list[float], str: float) -> None:
        super().__init__(pos, [0.0, 0.0, 0.0])
        self.color = np.array(color, dtype=np.float32)
        self.str = str
    
class Camera(Entity):
    # Third person camera that follows the player
    slots = ('forwards', 'right', 'up')
    def __init__(self, pos: list[float]) -> None:
        super().__init__(pos, [0.0, 0.0, 0.0])
        self.update(0)
    
    def update(self, delta_time: float):
        theta = self.rot[2]
        phi = self.rot[1]
        
        self.forwards = np.array([
            np.cos(theta) * np.cos(phi),
            np.sin(theta) * np.cos(phi),
            np.sin(phi)
        ], dtype=np.float32)
        
        self.right = np.cross(self.forwards, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        
        self.up = np.cross(self.right, self.forwards)
    
    def get_view_matrix(self):
        view_matrix = np.identity(4, dtype=np.float32)
        view_matrix[0, 0:3] = self.right
        view_matrix[1, 0:3] = self.up
        view_matrix[2, 0:3] = self.forwards
        view_matrix[0:3, 3] = -self.pos
        return view_matrix
    
    def move(self, d_pos):
        self.pos += d_pos[0] * self.forwards + d_pos[1] * self.right + d_pos[2] * self.up
        self.pos[2] = 2.0
    
    def rotate(self, d_rot):
        self.rot += d_rot
        self.rot[0] %= 360
        self.rot[1] = min(89.9, max(-89.9, self.rot[1]))
        self.rot[2] %= 360                  

class Scene:
    def __init__(self) -> None:
        self.entities = {
            "car": [
                Car(pos=[6, 0, 0], rot=[0, 0, 0]),
            ],
            "point_lights": [
                PointLight(pos=[4, 0, 2], color=[1, 1, 1], str=1.0)
            ]
        }
        self.player = Camera(pos=[0, 0, 2])
    
    def update(self, delta_time: float):
        for entity in self.entities.values():
            for e in entity:
                e.update(delta_time)
        self.player.update(delta_time)
    
    def move_player(self, d_pos: list[float]):
        self.player.move(d_pos)
    
    def rotate_player(self, d_rot: list[float]):
        self.player.rotate(d_rot)

class Material:
    def __init__(self, filepath) -> None:
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        # set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        image = pg.image.load(filepath).convert()
        width, height = image.get_rect().size
        image_data = pg.image.tostring(image, 'RGBA')
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
        glGenerateMipmap(GL_TEXTURE_2D)
        
    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        
    def destroy(self):
        glDeleteTextures(1, (self.texture,))
       
class Mesh:
    slots = ('vbo', 'vao', 'vertex_count')
    def __init__(self, filepath) -> None:
        vertices = loadMesh(filepath)
        self.vertex_count = len(vertices)
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
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))
    
    
if __name__ == "__main__":
    app = App()
    app.run()
    app.quit()