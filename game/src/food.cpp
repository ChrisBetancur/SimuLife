#include <food.h>

Food::Food(int x, int y) : Sprite(x, y, RED, FOOD) {}

Food::~Food() {}

bool Food::move(int dx, int dy) {
    // Food does not move
}

void Food::draw(SDL_Renderer* renderer) {

    // Draw food as a filled circle
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // Red color
    drawCircle(renderer, x, y, FOOD_SIZE, true);
}