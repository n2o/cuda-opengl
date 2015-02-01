#include <GL/glew.h>
#include <GL/freeglut.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "complexRange.h"
#include "mandel_GPU.cuh"


CRange range;

dim3 windowSize(1024,512);

static GLuint pbo_buffer = 0;
struct cudaGraphicsResource *cuda_pbo_resource;

/**
 * Bind OpenGL Buffer to CUDA
 */
void initGL(void) {
    // Generate OpenGL Pixel Buffer Object
    glGenBuffers(1, &pbo_buffer);
    // Bind this buffer to set the correct state
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_buffer);
    // Allocate memory
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, sizeof(uchar4) * windowSize.x * windowSize.y, 0, GL_DYNAMIC_DRAW_ARB);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_buffer, cudaGraphicsMapFlagsWriteDiscard));
}

/**
 * Draw in window
 */
void displayForGlut(void) {
    // Clears the pixels
    glClear(GL_COLOR_BUFFER_BIT);

    // Draw a square with specified color and the 4 points
    glColor3f(0, 0, 255);
    glBegin(GL_QUADS);
    glVertex3f(0.10, 0.10, 0.0);
    glVertex3f(0.9, 0.10, 0.0);
    glVertex3f(0.9, 0.9, 0.0);
    glVertex3f(0.10, 0.9, 0.0);
    glEnd();

    // Draws one point
    glColor3f(0, 255, 0);
    glBegin(GL_POINTS);
    glVertex3f(0.50, 0.30, 0.0);
    glEnd();

    glFlush();
}

/**
 * Initialize window
 */
int initGlutDisplay(int argc, char* argv[]) {
    // Init window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB);
    glutInitWindowSize(windowSize.x, windowSize.y);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Jeah OpenGL");

    // Set perspective
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);

    glewInit();

    calcGrid(windowSize);
    range.set(-2.0, -1.0, 2.0, 1.0, windowSize);

    glutDisplayFunc(displayForGlut);
    glutMainLoop();

    return 0;
}

int main(int argc, char* argv[]) {
    initGlutDisplay(argc, argv);

    return 0;
}
