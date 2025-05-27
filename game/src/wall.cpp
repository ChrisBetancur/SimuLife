#include <wall.h>

Wall::Wall(int x, int y) : Sprite(x, y, BLACK, WALL) {}

Wall::~Wall() {}

bool Wall::move(int dx, int dy) {
    // Wall does not move
}

void Wall::draw(SDL_Renderer* renderer) {
    // Draw wall as a filled rectangle
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // Black color
    SDL_Rect rect = {x - WALL_WIDTH / 2, y - WALL_HEIGHT / 2, WALL_WIDTH, WALL_HEIGHT};
    SDL_RenderFillRect(renderer, &rect);
}