#ifndef FOOD_H
#define FOOD_H
#include <SDL.h>
#include <iostream>
#include <sprites.h>

#define FOOD_SIZE 1

class Food : public Sprite {
    public:
        Food(int x, int y);

        ~Food() override;

        bool move(int dx, int dy) override;

        void draw(SDL_Renderer* renderer) override;
};

#endif