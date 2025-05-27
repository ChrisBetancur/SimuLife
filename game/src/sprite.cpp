#include <sprites.h>

void drawCircle(SDL_Renderer* renderer, int x_c, int y_c, int r, bool filled) {
    int x = 0;
    int y = r;
    int d = 1 - r;  // Initial decision parameter

    while (x <= y) {
        // Eight symmetric points
        SDL_RenderDrawPoint(renderer, x_c + x, y_c + y);
        SDL_RenderDrawPoint(renderer, x_c - x, y_c + y);
        SDL_RenderDrawPoint(renderer, x_c + x, y_c - y);
        SDL_RenderDrawPoint(renderer, x_c - x, y_c - y);
        SDL_RenderDrawPoint(renderer, x_c + y, y_c + x);
        SDL_RenderDrawPoint(renderer, x_c - y, y_c + x);
        SDL_RenderDrawPoint(renderer, x_c + y, y_c - x);
        SDL_RenderDrawPoint(renderer, x_c - y, y_c - x);

        // If filled, draw horizontal lines between the points
        if (filled) {
            SDL_RenderDrawLine(renderer, x_c - x, y_c + y, x_c + x, y_c + y);
            SDL_RenderDrawLine(renderer, x_c - x, y_c - y, x_c + x, y_c - y);
            SDL_RenderDrawLine(renderer, x_c - y, y_c + x, x_c + y, y_c + x);
            SDL_RenderDrawLine(renderer, x_c - y, y_c - x, x_c + y, y_c - x);
        }

        if (d < 0) {
            d = d + 2*x + 3;
        } else {
            d = d + 2*(x - y) + 5;
            y--;
        }
        x++;
    }
}


Sprite::Sprite(int x, int y, Color color, Type type) : x(x), y(y), color(color), m_type(type) {}

Sprite::~Sprite() {}


void Sprite::setPosition(int newX, int newY) {
    x = newX;
    y = newY;
}

void Sprite::getPosition(int &outX, int &outY) const {
    outX = x;
    outY = y;
} 

void Sprite::getColor(Color &outColor) const {
    outColor = color;
}

void Sprite::setColor(Color newColor) {
    color = newColor;
}

Sprite::Type Sprite::getType() const {
    return m_type;
}