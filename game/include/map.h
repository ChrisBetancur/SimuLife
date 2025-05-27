#ifndef MAP_H
#define MAP_H
#include <SDL.h>
#include <iostream>
#include <sprites.h>
#include <food.h>
#include <organism.h>
#include <wall.h>

#define CELL_SIZE 100 // Size of each cell in the grid

enum CellType {
    EMPTY = 0,
    WALL = 1,
    FOOD = 2,
    ORGANISM = 3
};

class Map {
    private:
        int width, height; // Dimensions of the map
        Sprite*** grid; // 2D array to represent the map
        int food_count = 0;
        

    public:

        Map(int w, int h);

        void reset();

        ~Map();

        void addOrganism(int x, int y, Genome genome);

        int getWidth() const;

        int getHeight() const;

        void organismCollisionFood(Sprite* sprite_org);

        bool isWall(int x, int y) const;

        void draw_map(SDL_Renderer* renderer);

        std::vector<CellType> getVision(int x, int y, Direction facing, int depth, int org_size) const;
};

#endif