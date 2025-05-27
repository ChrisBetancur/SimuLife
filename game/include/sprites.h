#ifndef SPRITES_H
#define SPRITES_H

#include <SDL.h>
#include <iostream>

enum Color {
    RED,
    GREEN,
    BLUE,
    YELLOW,
    CYAN,
    MAGENTA,
    WHITE,
    BLACK
};

// USED IN REINFORCEMENT LEARNING
enum Direction {
    UP=0,
    DOWN=1,
    LEFT=2,
    RIGHT=3
};

void drawCircle(SDL_Renderer* renderer, int x_c, int y_c, int r, bool filled);

class Sprite {
    protected:
        int x, y; // Position
        Color color; // Color
        enum Type {
            EMPTY = 0,
            WALL = 1,
            FOOD = 2,
            ORGANISM = 3
        } type;

        Type m_type;

    public: 

        Sprite(int x, int y, Color color, Type type);

        virtual ~Sprite();

        virtual bool move(int dx, int dy) = 0;

        virtual void draw(SDL_Renderer* renderer) = 0;

        void setPosition(int newX, int newY);

        void getPosition(int &outX, int &outY) const;

        void getColor(Color &outColor) const;

        void setColor(Color newColor);

        Type getType() const;
};

#endif