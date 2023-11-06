import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
import random

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


class Player:
    pass
class Scene:
    pass
class Light:
    pass

class App:
    def __init__(self):
        #initialise pygame
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                    pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode((640,480), pg.OPENGL|pg.DOUBLEBUF)
        pg.mouse.set_pos((320,240))
        pg.mouse.set_visible(False)

        self.scene = Scene()

        self.engine = Engine(self.scene)

        self.run()
    
    def run(self):
        running = True
        while (running):
            #check events
            for event in pg.event.get():
                if (event.type == pg.KEYDOWN and event.key==pg.K_ESCAPE):
                    running = False
            self.handleKeys()
            #update objects
            self.scene.update()
            #refresh screen
            self.engine.draw(self.scene)
            self.showFrameRate()
        self.quit()

    def handleKeys(self):
        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT]:
            pass
        if keys[pg.K_RIGHT]:
            pass
        if keys[pg.K_UP]:
            pass
        if keys[pg.K_DOWN]:
            pass

    def showFrameRate(self):
        self.currentTime = pg.time.get_ticks()
        delta = self.currentTime - self.lastTime
        if (delta >= 1000):
            framerate = int(1000.0 * self.numFrames/delta)
            pg.display.set_caption(f"Running at {framerate} fps.")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0 / framerate)
        self.numFrames += 1
    
    def quit(self):

        self.engine.quit()
        pg.quit()

class Engine:
    pass

class Material:
    pass

class Mesh:
    pass

class Texture:
    pass

if __name__ == "__main__":
    app = App()
    app.run()
    app.quit()