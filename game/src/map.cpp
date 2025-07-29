#include <map.h>
#include <random>

Map::Map(int w, int h) : width(w), height(h) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 9999);


    grid = new Sprite**[height];

    int p = 15; // percentage of food in the map
    int threshold = int( (p/100.0) * 10000 );


    for (int i = 0; i < height; ++i) {
        grid[i] = new Sprite*[width];
        for (int j = 0; j < width; ++j) {
            if (i == 0 || i == height - 1 || j == 0 || j == width - 1) {
                grid[i][j] = new Wall(j, i); // Set borders as walls
            }
            else if (distrib(gen) < threshold) { // Randomly place food
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
            else if (distrib(gen) < 2) { // Randomly place food
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

std::vector<CellType> Map::getVision(int x, int y, Direction facing, int depth, int org_size) const {
    std::vector<CellType> cells;
    int dx = 0, dy = 0;

    switch (facing) {
        case UP:
            dy = -1;
            break;
        case DOWN:  dy = +1; break;
        case LEFT:  dx = -1; break;
        case RIGHT: dx = +1; break;
    }

    for (int i = 0; i < depth + 1; ++i) {
        for (int j = CELL_SIZE * i; j < CELL_SIZE * (i + 1); ++j) {
            int newX = x + dx * j;
            int newY = y + dy * j;

            int x_min = 0;
            int x_max = 0;

            int y_min = 0;
            int y_max = 0;

            int scope = org_size;

            switch (facing) {
                case UP:
                    x_min = x - scope;
                    x_max = x + scope;

                    // reduce min and max if out of bounds
                    if (x_min < 0) {
                        x_min = 0;
                    }
                    if (x_max >= width) {
                        x_max = width - 1;
                    }
                    break;
                case DOWN:  
                    x_min = x - scope;
                    x_max = x + scope;

                    // reduce min and max if out of bounds
                    if (x_min < 0) {
                        x_min = 0;
                    }
                    if (x_max >= width) {
                        x_max = width - 1;
                    }
                    break;
                case LEFT:
                    y_min = y - scope;
                    y_max = y + scope;

                    // reduce min and max if out of bounds
                    if (y_min < 0) {
                        y_min = 0;
                    }
                    if (y_max >= height) {
                        y_max = height - 1;
                    }
                    break;
                case RIGHT: 
                    y_min = y - scope;
                    y_max = y + scope;

                    // reduce min and max if out of bounds
                    if (y_min < 0) {
                        y_min = 0;
                    }
                    if (y_max >= height) {
                        y_max = height - 1;
                    }
                    break;
            }


            if (x_min != 0) {
                for (int k = x_min; k < x_max; ++k) {


                    // Check for wall, cannot see beyond wall
                    if (grid[newY][k] != nullptr && grid[newY][k]->getType() == WALL) {
                        cells.push_back(WALL);
                        return cells;
                    }
                    // Check for food
                    if (grid[newY][k] != nullptr && grid[newY][k]->getType() == FOOD) {

                        cells.push_back(FOOD);
                        continue;
                    }
                    // Check for organism
                    if (grid[newY][k] != nullptr && grid[newY][k]->getType() == ORGANISM) {
                        cells.push_back(ORGANISM);
                        continue;
                    }

                }
            }
            else {
                for (int k = y_min; k < y_max; ++k) {
                    // Check for wall, cannot see beyond the wall
                    if (grid[k][newX] != nullptr && grid[k][newX]->getType() == WALL) {
                        cells.push_back(WALL);

                        return cells;
                    }
                    // Check for food
                    if (grid[k][newX] != nullptr && grid[k][newX]->getType() == FOOD) {
                        cells.push_back(FOOD);
                        continue;
                    }
                    // Check for organism
                    if (grid[k][newX] != nullptr && grid[k][newX]->getType() == ORGANISM) {
                        cells.push_back(ORGANISM);
                        continue;
                    }
                    // Check for empty
                    /*if (grid[k][newX] == nullptr) {
                        cells.push_back(EMPTY);
                        continue;
                    }*/
    
                
                }
            }
        }
    }

    return cells;
}

std::vector<double> Map::getFoodCounts() const {
    std::vector<double> food_counts; // 3x3 grid around the organism

    food_counts.resize(9, 0); // Initialize with 0s

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // given the width and height of the map, calculate the sector
            int sector_x = j / (width / 3);
            int sector_y = i / (height / 3);
            int sector_index = sector_y * 3 + sector_x;
            
            if (grid[i][j] != nullptr && grid[i][j]->getType() == FOOD) {
                food_counts[sector_index]++;
            }

        }
    }


    return food_counts;
}