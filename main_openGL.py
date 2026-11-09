import os
import random
import numpy as np
import pyglet
from pyglet.gl import *

class main_application(pyglet.window.Window):
    def __init__(self, width, height):
        super().__init__(caption='Game_of_Life', width=width, height=height)
        self.set_icon(pyglet.image.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'icon.png')))

        self.height = height
        self.width = width

        self.fill_color = (255, 255, 255)
        self.stop = False
        self.tile_size = 1
        self.speed = 0.001

        self.x_offset = 0
        self.y_offset = 0

        self.cells = np.zeros((self.height // self.tile_size, self.width // self.tile_size), dtype=np.uint8)
        self.draw_cells = {}

        self.figures_batch = pyglet.graphics.Batch()
        self.vertex_list = None

        self.get_cells()

    def get_cells(self):
        id = 0
        self.cells = np.zeros((self.height // self.tile_size, self.width // self.tile_size), dtype=np.uint8)
        self.cells.fill(0)
        self.draw_cells = {}
        for y in range(self.height // self.tile_size):
            for x in range(self.width // self.tile_size):
                if random.random() < random.uniform(0.1, 0.01):
                    self.cells[y, x] = 1
                    self.draw_cells[id] = [y, x]
                id += 1

    def animation(self, dt):
        # Calculate the number of neighbors for each cell using a 3x3 neighborhood sum filter
        neighbors_count = np.zeros_like(self.cells, dtype=np.uint8)
        neighbors_count += np.roll(self.cells, (-1, -1), axis=(0, 1))
        neighbors_count += np.roll(self.cells, (-1, 0), axis=(0, 1))
        neighbors_count += np.roll(self.cells, (-1, 1), axis=(0, 1))
        neighbors_count += np.roll(self.cells, (0, -1), axis=(0, 1))
        neighbors_count += np.roll(self.cells, (0, 1), axis=(0, 1))
        neighbors_count += np.roll(self.cells, (1, -1), axis=(0, 1))
        neighbors_count += np.roll(self.cells, (1, 0), axis=(0, 1))
        neighbors_count += np.roll(self.cells, (1, 1), axis=(0, 1))

        # Apply the Game of Life rules to update the cells
        survive_mask = np.logical_and(self.cells == 1, np.logical_or(neighbors_count == 2, neighbors_count == 3))
        birth_mask = np.logical_and(self.cells == 0, neighbors_count == 3)
        self.cells = np.logical_or(survive_mask, birth_mask).astype(np.uint8)

        # Update the draw_cells dictionary with the new live cells
        self.draw_cells = {}
        indices = np.argwhere(self.cells == 1)
        for id, (y, x) in enumerate(indices):
            self.draw_cells[id] = [y, x]


    def on_draw(self):
        self.clear()

        vertices = []
        colors = []

        for cell in self.draw_cells.values():
            x = (self.x_offset + cell[1]) * self.tile_size
            y = (self.y_offset + cell[0]) * self.tile_size
            vertices.extend([x, y, x + self.tile_size, y, x + self.tile_size, y + self.tile_size, x, y + self.tile_size])
            colors.extend([*self.fill_color] * 4)

        num_vertices = len(vertices) // 2

        if num_vertices > 0:
            vertex_format = 'v2f'
            color_format = 'c3B'

            if self.vertex_list is not None:
                self.vertex_list.resize(num_vertices)
                self.vertex_list.vertices = vertices
                self.vertex_list.colors = colors
            else:
                self.vertex_list = pyglet.graphics.vertex_list(num_vertices, (vertex_format, vertices), (color_format, colors))

            self.vertex_list.draw(pyglet.gl.GL_QUADS)


    def on_key_press(self, symbol, modifiers):
        # Stop sim
        if symbol == pyglet.window.key.E:
            if not self.stop:
                pyglet.clock.unschedule(self.animation)
                self.stop = True
            else:
                self.stop = False
                pyglet.clock.schedule_interval(self.animation, self.speed)

        # Speed up sim
        if symbol == pyglet.window.key.PLUS:
            self.tile_size += 1

        # Speed down sim
        if symbol == pyglet.window.key.MINUS:
            if self.tile_size >= 2:
                self.tile_size -= 1

        # Reset
        if symbol == pyglet.window.key.R:
            self.x_offset = 0
            self.y_offset = 0
            self.get_cells()

        # Exit
        if symbol == pyglet.window.key.ESCAPE:
            pyglet.app.exit()

        if symbol == pyglet.window.key.F11:
            window.set_fullscreen(not window.fullscreen)
        
        if symbol == pyglet.window.key.RIGHT:
            self.x_offset -= self.tile_size
        if symbol == pyglet.window.key.LEFT:
            self.x_offset += self.tile_size
        if symbol == pyglet.window.key.UP:
            self.y_offset -= self.tile_size
        if symbol == pyglet.window.key.DOWN:
            self.y_offset += self.tile_size

if __name__ == "__main__":

    scale = 1.5
    height = int(1080 * scale)
    width = int(1920 * scale)

    window = main_application(width=width, height=height)
    pyglet.clock.schedule_interval(window.animation, window.speed)
    

    @window.event
    def on_close():
        pyglet.app.exit()

    pyglet.app.run()
