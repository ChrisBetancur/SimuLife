#ifndef WALL_H
#define WALL_H
#include <SDL.h>
#include <iostream>

#include <sprites.h>

#define WALL_WIDTH 10
#define WALL_HEIGHT 10

class Wall : public Sprite {
    public:
        Wall(int x, int y);
        
        ~Wall() override;

        bool move(int dx, int dy) override;

        void draw(SDL_Renderer* renderer) override;
};

#endif