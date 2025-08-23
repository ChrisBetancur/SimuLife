#include <map.h>
#include <random>
#include <tuple>
#include <iostream>

Map::Map(int w, int h) : width(w), height(h) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 9999);


    grid = new Sprite**[height];

    int p = 20; // percentage of food in the map
    int threshold = int( (p/100.0) * 10000 );


    for (int i = 0; i < height; ++i) {
        grid[i] = new Sprite*[width];
        for (int j = 0; j < width; ++j) {
            if (i == 0 || i == height - 1 || j == 0 || j == width - 1) {
                grid[i][j] = new Wall(j, i); // Set borders as walls
            }
            else if (distrib(gen) < 100) { // Randomly place food
                grid[i][j] = new Food(j, i);
                food_count++;
            }
            else {
                grid[i][j] = nullptr; // Empty cell
            }

        }
    }
}

void Map::reset() {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            delete grid[i][j];  // Delete individual Sprite objects
        }
        delete[] grid[i];  // Delete row array
    }
    delete[] grid;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 9999);

    grid = new Sprite**[height];
    for (int i = 0; i < height; ++i) {
        grid[i] = new Sprite*[width];
        for (int j = 0; j < width; ++j) {
            if (i == 0 || i == height - 1 || j == 0 || j == width - 1) {
                grid[i][j] = new Wall(j, i); // Set borders as walls
            }
            else if (distrib(gen) < 6) { // Randomly place food
                grid[i][j] = new Food(j, i);
                food_count++;
            }
            else {
                grid[i][j] = nullptr; // Empty cell
            }

        }
    }
}

Map::~Map() {
    for (int i = 0; i < height; ++i) {
        delete[] grid[i];
    }
    delete[] grid;
}

void Map::addOrganism(int x, int y, Genome genome) {
    if (grid[y][x] == nullptr) {
        grid[y][x] = new Organism(x, y, genome);
    }
}


int Map::getWidth() const {
    return width;
}
int Map::getHeight() const {
    return height;
}

void Map::organismCollisionFood(Sprite* sprite_org) {
    Organism* organism = static_cast<Organism*>(sprite_org);
    int orgX, orgY, orgRadius;
    organism->getPosition(orgX, orgY);
    orgRadius = organism->getGenome().size;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (grid[i][j] && grid[i][j]->getType() == FOOD) {
                Food* food = static_cast<Food*>(grid[i][j]);
                int foodX, foodY, foodRadius;

                food->getPosition(foodX, foodY);
                foodRadius = FOOD_SIZE;

                // Calculate distance between centers
                int dx = orgX - foodX;
                int dy = orgY - foodY;
                int distanceSq = dx * dx + dy * dy;

                // Compare with sum of radii
                int radiusSum = orgRadius + foodRadius;
                if (distanceSq <= radiusSum * radiusSum) {
                    // Collision detected!
                    delete grid[i][j];
                    grid[i][j] = nullptr;
                    food_count--;
                    organism->eat();
                    eating = true; // Set eating flag
                }
            }
        }
    }
}

bool Map::isWall(int x, int y) const {
    // Out‑of‑bounds counts as a wall
    if (x < 0 || x >= width || y < 0 || y >= height)
        return true;
    Sprite* s = grid[y][x];
    return (s != nullptr && s->getType() == WALL);
}

int Map::getWallPosX(int x, int y) const {
    if (x < 0 || x >= width || y < 0 || y >= height)
        return -1; // Invalid position
    Sprite* s = grid[y][x];
    if (s != nullptr && s->getType() == WALL) {
        return x;
    }
    return -1; // No wall at this position
}

int Map::getWallPosY(int x, int y) const {
    if (x < 0 || x >= width || y < 0 || y >= height)
        return -1; // Invalid position
    Sprite* s = grid[y][x];
    if (s != nullptr && s->getType() == WALL) {
        return y;
    }
    return -1; // No wall at this position
}

void Map::draw_map(SDL_Renderer* renderer) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            Sprite* cell = grid[i][j];
            if (cell != nullptr) {
                cell->draw(renderer);
            }
        }
    }
}

void Map::drawVision(SDL_Renderer* renderer) const {
    /*SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);

    int minX, minY, maxX, maxY;
    std::tie(minX, minY, maxX, maxY) = org_vision;

    // convert cell‐coords → pixel‐coords, and include xmax/ymax cell
    SDL_Rect rect {
        minX,
        minY,
        (maxX - minX) + 4,
        (maxY - minY) + 4
    };

    SDL_RenderDrawRect(renderer, &rect);*/
}

std::tuple<int, bool, int> Map::getVision(int x, int y,
                                        Direction facing,
                                        int depth,
                                        int org_size) const
{
    // compute deltas for facing
    int dx=0, dy=0;
    switch(facing) {
        case UP:    dy = -1; break;
        case DOWN:  dy = +1; break;
        case LEFT:  dx = -1; break;
        case RIGHT: dx = +1; break;
    }

    // half‐width perpendicular to facing
    int halfW = org_size;
    // total length forward
    int length = depth * org_size;

    // bounding box coords
    int xmin, xmax, ymin, ymax;
    if (dx > 0) {              // → RIGHT
        xmin = x + 1;
        xmax = x + length;
        ymin = y - halfW;
        ymax = y + halfW;
    }
    else if (dx < 0) {         // ← LEFT
        xmin = x - length;
        xmax = x - 1;
        ymin = y - halfW;
        ymax = y + halfW;
    }
    else if (dy > 0) {         // ↓ DOWN
        ymin = y + 1;
        ymax = y + length;
        xmin = x - halfW;
        xmax = x + halfW;
    }
    else {                     // ↑ UP
        ymin = y - length;
        ymax = y - 1;
        xmin = x - halfW;
        xmax = x + halfW;
    }

    org_vision = std::make_tuple(xmin, ymin, xmax, ymax);

    bool sawWall = false;
    int foodCount = 0;
    int wall_distance = -1; // distance to the first wall encountered, -1 if no wall

    // scan box
    for (int j = xmin; j <= xmax; ++j) {
        for (int k = ymin; k <= ymax; ++k) {
            if (j < 0 || j >= width || k < 0 || k >= height) {
                sawWall = true;
                if (wall_distance == -1) {
                    wall_distance = std::abs(j - x) + std::abs(k - y);
                }
                continue;
            }
            auto* cell = grid[k][j];
            if (!cell) continue;
            if (cell->getType() == WALL) {
                sawWall = true;
            } else if (cell->getType() == FOOD) {
                ++foodCount;
            }
        }
    }

    return std::make_tuple(foodCount, sawWall, wall_distance);
}

std::vector<double> Map::getFoodCounts() const {
    std::vector<double> food_counts; // 3x3 grid around the organism

    food_counts.resize(9, 0); // Initialize with 0s

    if (width == 0 || height == 0) return food_counts; // Handle empty map case

    // Calculate sector sizes safely
    const int sector_width = std::max(1, width / 3);
    const int sector_height = std::max(1, height / 3);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // Calculate sector with bounds checking
            int sector_x = std::min(j / sector_width, 2);
            int sector_y = std::min(i / sector_height, 2);
            int sector_index = sector_y * 3 + sector_x;
            
            if (sector_index >= 0 && sector_index < 9) {
                if (grid[i][j] != nullptr && grid[i][j]->getType() == FOOD) {
                    food_counts[sector_index]++;
                }
            }
        }
    }


    return food_counts;
}