"""Game of Life simulated on the GPU with bit-packed cells.

State texture (GL_R32UI, ping-pong): one texel holds 32 horizontally adjacent cells as bits,
texel (wx, y) bit b = cell (32 * wx + b, y). The step shader counts the 8 neighbours of all 32 cells
of a word at once with bitwise adders, so one shader invocation computes 32 cells.

Heat texture (GL_R8, ping-pong, one texel per cell): 1.0 while a cell is alive, fading afterwards.
It is updated once per displayed frame (not per generation) and drives the glowing trails.

The field is a torus. Any width works; bits beyond the field width in the last word stay 0.
Requires a current OpenGL 3.3+ context (any pyglet window).
"""
from ctypes import byref, c_uint32

import numpy as np
from pyglet.gl import (
    GL_CLAMP_TO_EDGE, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER, GL_FRAMEBUFFER_COMPLETE, GL_NEAREST,
    GL_PACK_ALIGNMENT, GL_R8, GL_R32UI, GL_RED, GL_RED_INTEGER, GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TRIANGLES,
    GL_UNPACK_ALIGNMENT, GL_UNSIGNED_BYTE, GL_UNSIGNED_INT, GLuint, glActiveTexture, glBindFramebuffer,
    glBindTexture, glCheckFramebufferStatus, glDeleteFramebuffers, glDeleteTextures,
    glFramebufferTexture2D, glGenFramebuffers, glGenTextures, glGetTexImage, glPixelStorei, glReadPixels,
    glTexImage2D, glTexParameteri, glTexSubImage2D, glViewport,
)
from pyglet.graphics.shader import Shader, ShaderProgram

FULLSCREEN_VERT = """#version 330 core
in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }
"""

STEP_FRAG = """#version 330 core
uniform usampler2D state;
uniform int width;              // field width in cells
layout(location = 0) out uint frag;

ivec2 size;

uint fetch(int wx, int y) {
    return texelFetch(state, ivec2((wx + size.x) % size.x, (y + size.y) % size.y), 0).r;
}

int valid_bits(int wx) {        // cells stored in word wx (the last word may be partial)
    return wx == size.x - 1 ? width - 32 * (size.x - 1) : 32;
}

// bit-sliced 3-bit counter: adds one bit-vector to (s0, s1, s2), counts above 7 wrap (8 -> 0 is fine)
void add(uint v, inout uint s0, inout uint s1, inout uint s2) {
    uint c0 = s0 & v;  s0 ^= v;
    uint c1 = s1 & c0; s1 ^= c0;
    s2 ^= c1;
}

void main() {
    size = textureSize(state, 0);
    ivec2 p = ivec2(gl_FragCoord.xy);
    uint vc = uint(valid_bits(p.x));
    uint vl = uint(valid_bits((p.x - 1 + size.x) % size.x));

    uint s0 = 0u, s1 = 0u, s2 = 0u, self_word = 0u;
    for (int dy = -1; dy <= 1; dy++) {
        uint C = fetch(p.x, p.y + dy);
        uint L = fetch(p.x - 1, p.y + dy);
        uint R = fetch(p.x + 1, p.y + dy);
        uint west = (C << 1) | ((L >> (vl - 1u)) & 1u);       // bit b = cell b-1
        uint east = (C >> 1) | ((R & 1u) << (vc - 1u));       // bit b = cell b+1
        add(west, s0, s1, s2);
        add(east, s0, s1, s2);
        if (dy != 0) add(C, s0, s1, s2); else self_word = C;
    }
    uint mask = vc == 32u ? 0xFFFFFFFFu : ((1u << vc) - 1u);
    // alive next: count == 3, or count == 2 and alive now
    frag = (s1 & ~s2 & (s0 | self_word)) & mask;
}
"""

HEAT_FRAG = """#version 330 core
uniform usampler2D state;
uniform sampler2D heat;
uniform float decay;
out vec4 frag;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    uint word = texelFetch(state, ivec2(p.x >> 5, p.y), 0).r;
    bool alive = ((word >> uint(p.x & 31)) & 1u) == 1u;
    float h = texelFetch(heat, p, 0).r;
    // subtract a little extra so 8-bit rounding cannot keep the heat stuck above zero
    frag = vec4(alive ? 1.0 : max(h * decay - 1.5 / 255.0, 0.0), 0.0, 0.0, 1.0);
}
"""

# Two triangles covering the whole viewport.
FULLSCREEN_QUAD = (-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1)


def fullscreen_quad(program: ShaderProgram):
    return program.vertex_list(6, GL_TRIANGLES, position=('f', FULLSCREEN_QUAD))


def pack(cells: np.ndarray, words: int) -> np.ndarray:
    """(H, W) 0/1 array -> (H, words) uint32, bit b of word wx = cell 32*wx + b."""
    h, w = cells.shape
    padded = np.zeros((h, words * 32), np.uint8)
    padded[:, :w] = cells > 0
    return np.packbits(padded, axis=1, bitorder="little").view("<u4")


def unpack(words: np.ndarray, width: int) -> np.ndarray:
    bits = np.unpackbits(np.ascontiguousarray(words, "<u4").view(np.uint8), axis=1, bitorder="little")
    return bits[:, :width]


def _texture(internal, fmt, typ, w, h):
    tex = GLuint()
    glGenTextures(1, byref(tex))
    glBindTexture(GL_TEXTURE_2D, tex)
    for param, value in ((GL_TEXTURE_MIN_FILTER, GL_NEAREST), (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
                         (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)):
        glTexParameteri(GL_TEXTURE_2D, param, value)
    glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, typ, None)
    return tex


def _framebuffer(tex):
    fbo = GLuint()
    glGenFramebuffers(1, byref(fbo))
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    if status != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"Framebuffer incomplete (status 0x{status:x})")
    return fbo


class GpuLife:
    def __init__(self, width: int, height: int, decay: float = 0.9):
        self.width = width
        self.height = height
        self.words = (width + 31) // 32
        self.decay = decay

        self._step = ShaderProgram(Shader(FULLSCREEN_VERT, 'vertex'), Shader(STEP_FRAG, 'fragment'))
        self._heat = ShaderProgram(Shader(FULLSCREEN_VERT, 'vertex'), Shader(HEAT_FRAG, 'fragment'))
        self._step_quad = fullscreen_quad(self._step)
        self._heat_quad = fullscreen_quad(self._heat)

        self._state = [_texture(GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, self.words, height) for _ in range(2)]
        self._state_fbo = [_framebuffer(t) for t in self._state]
        self._heat_tex = [_texture(GL_R8, GL_RED, GL_UNSIGNED_BYTE, width, height) for _ in range(2)]
        self._heat_fbo = [_framebuffer(t) for t in self._heat_tex]
        self._cur = 0        # index of the current state texture
        self._heat_cur = 0

    def delete(self):
        fbos = self._state_fbo + self._heat_fbo
        texs = self._state + self._heat_tex
        glDeleteFramebuffers(len(fbos), (GLuint * len(fbos))(*fbos))
        glDeleteTextures(len(texs), (GLuint * len(texs))(*texs))
        self._step_quad.delete()
        self._heat_quad.delete()

    # --- state -------------------------------------------------------------------------------

    @property
    def state_texture(self) -> GLuint:
        """GL_R32UI texture with the current generation, 32 cells per texel."""
        return self._state[self._cur]

    @property
    def heat_texture(self) -> GLuint:
        """GL_R8 texture with one heat value per cell (for trails)."""
        return self._heat_tex[self._heat_cur]

    def upload(self, cells: np.ndarray):
        """Replace the whole field. ``cells`` has shape (height, width), row 0 = bottom."""
        assert cells.shape == (self.height, self.width)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        packed = np.ascontiguousarray(pack(cells, self.words))
        glBindTexture(GL_TEXTURE_2D, self.state_texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.words, self.height, GL_RED_INTEGER, GL_UNSIGNED_INT,
                        packed.ctypes.data)
        heat = np.ascontiguousarray((cells > 0).astype(np.uint8) * np.uint8(255))
        glBindTexture(GL_TEXTURE_2D, self.heat_texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RED, GL_UNSIGNED_BYTE,
                        heat.ctypes.data)

    def set_cells(self, cells, alive: bool):
        """Set or clear a collection of (x, y) cells (coordinates wrap around)."""
        by_word = {}
        for x, y in cells:
            x %= self.width
            y %= self.height
            by_word.setdefault((x >> 5, y), []).append(x & 31)
        if not by_word:
            return
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._state_fbo[self._cur])
        for (wx, y), bits in by_word.items():
            word = c_uint32()
            glReadPixels(wx, y, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, byref(word))
            value = word.value
            for b in bits:
                value = value | (1 << b) if alive else value & ~(1 << b)
            word.value = value
            glBindTexture(GL_TEXTURE_2D, self.state_texture)
            glTexSubImage2D(GL_TEXTURE_2D, 0, wx, y, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, byref(word))
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, self.heat_texture)
        one = np.array([255 if alive else 0], np.uint8)
        for x, y in cells:
            glTexSubImage2D(GL_TEXTURE_2D, 0, x % self.width, y % self.height, 1, 1, GL_RED, GL_UNSIGNED_BYTE,
                            one.ctypes.data)

    def set_cell(self, x: int, y: int, alive: bool):
        self.set_cells([(x, y)], alive)

    def read(self) -> np.ndarray:
        """Download the current field as 0/1 array of shape (height, width)."""
        out = np.empty((self.height, self.words), np.uint32)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glBindTexture(GL_TEXTURE_2D, self.state_texture)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, out.ctypes.data)
        return unpack(out, self.width)

    def read_heat(self) -> np.ndarray:
        out = np.empty((self.height, self.width), np.uint8)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glBindTexture(GL_TEXTURE_2D, self.heat_texture)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_BYTE, out.ctypes.data)
        return out

    # --- simulation --------------------------------------------------------------------------

    def step(self, generations: int = 1):
        """Advance the simulation. Leaves the default framebuffer bound; the caller restores its viewport."""
        if generations <= 0:
            return
        glViewport(0, 0, self.words, self.height)
        glActiveTexture(GL_TEXTURE0)
        self._step.use()
        self._step['state'] = 0
        self._step['width'] = self.width
        for _ in range(generations):
            target = 1 - self._cur
            glBindFramebuffer(GL_FRAMEBUFFER, self._state_fbo[target])
            glBindTexture(GL_TEXTURE_2D, self._state[self._cur])
            self._step_quad.draw(GL_TRIANGLES)
            self._cur = target
        self._step.stop()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def update_heat(self):
        """One fade step of the trails (call once per displayed frame)."""
        target = 1 - self._heat_cur
        glViewport(0, 0, self.width, self.height)
        glBindFramebuffer(GL_FRAMEBUFFER, self._heat_fbo[target])
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.state_texture)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self._heat_tex[self._heat_cur])
        self._heat.use()
        self._heat['state'] = 0
        self._heat['heat'] = 1
        self._heat['decay'] = self.decay
        self._heat_quad.draw(GL_TRIANGLES)
        self._heat.stop()
        glActiveTexture(GL_TEXTURE0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self._heat_cur = target
