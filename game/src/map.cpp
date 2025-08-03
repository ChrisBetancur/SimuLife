#include <map.h>
#include <random>

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

    int food_min = 4; // Minimum food count to consider a sector

    for (int i = 0; i < depth; ++i) {

        CellType cell_type = EMPTY;

        int food_count = 0;

        for (int j = 0; j < org_size; ++j) {
            for (int k = 0; k < CELL_SIZE; ++k) {
                int newX = x + dx * (i * org_size + j) + k;
                int newY = y + dy * (i * org_size + j) + k;

                if (newX < 0 || newX >= width || newY < 0 || newY >= height) {
                    cell_type = WALL; // Out of bounds is considered a wall
                    break;
                }

                if (grid[newY][newX] != nullptr) {
                    if (grid[newY][newX]->getType() == WALL && food_count < food_min) {
                        cell_type = WALL;
                    }
                    else {
                        cell_type = FOOD;
                        if (cell_type == FOOD) {
                            food_count++;
                        }
                    }
                }

            }
        }

        cells.push_back(cell_type);  
    }

    // print cells size
    std::cout << "Cells size: " << cells.size() << std::endl;
    // print cells content, dont print number, print the enum name
    for (const auto& cell : cells) {
        switch (cell) {
            case EMPTY:     std::cout << "EMPTY "; break;
            case WALL:      std::cout << "WALL "; break;
            case FOOD:      std::cout << "FOOD "; break;
            case ORGANISM:  std::cout << "ORGANISM "; break;
        }
    }

    std::string direction_name;
    switch (facing) {
        case UP:    direction_name = "UP"; break;
        case DOWN:  direction_name = "DOWN"; break;
        case LEFT:  direction_name = "LEFT"; break;
        case RIGHT: direction_name = "RIGHT"; break;
    }
    std::cout << "Vision for direction " << direction_name << std::endl;

    return cells;
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