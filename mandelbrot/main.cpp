#include <GL/gl.h>
#include <GL/glut.h>

void displayForGlut(void) {
    // clears the pixels
    glClear(GL_COLOR_BUFFER_BIT);

    // draw a square with specified color and the 4 points
    glColor3f(0, 0, 255);
    glBegin(GL_QUADS);
    glVertex3f(0.10, 0.10, 0.0);
    glVertex3f(0.9, 0.10, 0.0);
    glVertex3f(0.9, 0.9, 0.0);
    glVertex3f(0.10, 0.9, 0.0);
    glEnd();

    // draws one point
    glColor3f(0, 255, 0);
    glBegin(GL_POINTS);
    glVertex3f(0.50, 0.30, 0.0);
    glEnd();

    glFlush();
}

int initGlutDisplay(int argc, char* argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB);
    glutInitWindowSize(1024, 512);
    glutInitWindowPosition(100, 100);

    glutCreateWindow("Jeah OpenGL");

    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);

    glutDisplayFunc(displayForGlut);
    glutMainLoop();

    return 0;
}

int main(int argc, char* argv[]) {
    initGlutDisplay(argc, argv);

    return 0;
}
