#include <organism.h>

Organism::Organism(int x, int y, Genome genome) : 
    Sprite(x, y, color, FOOD),
    m_genome(genome),
    basal_rate(0.5f),
    speed_coeff(0.01f),
    starve_thresh(0.3f),
    starve_accel(1.0f),
    original_size(genome.size){
    max_energy_lvl = energy_lvl = 100.0f;
}

void Organism::reset(int x, int y) {
    // Reset the organism's position and energy level
    this->x = x;
    this->y = y;
    energy_lvl = max_energy_lvl;
    foods_eaten = 0;
    m_genome.size = original_size;
}

void Organism::applyEnergyCost() {
    // Base cost per move
    float drain = basal_rate;
    // Extra cost proportional to speed (i.e. cells moved)
    drain += speed_coeff * m_genome.speed;

    // Optional: accelerated starvation when low on energy
    float frac = energy_lvl / max_energy_lvl;
    if (frac < starve_thresh) {
        float alpha = (starve_thresh - frac) / starve_thresh;
        drain *= (1.0f + alpha * (starve_accel - 1.0f));
    }

    energy_lvl -= drain;
    if (energy_lvl < 0) energy_lvl = 0;
}
        
bool Organism::move(int dx, int dy) {

    if (energy_lvl <= 0) {
        return false;
    }

    x += dx * m_genome.speed;
    y += dy * m_genome.speed;

    // if dx and dy are both 0, then we will apply the energy cost
    applyEnergyCost();

    if      (dx>0)  m_direction = RIGHT;
    else if (dx<0)  m_direction = LEFT;
    else if (dy>0)  m_direction = DOWN;
    else if (dy<0)  m_direction = UP;

    return true;
}

void Organism::draw(SDL_Renderer* renderer) {
    // Draw the organism as a filled circle
    // color based on gender

    if (m_genome.gender == MALE) {
        SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255); // Blue color
    } else {
        SDL_SetRenderDrawColor(renderer, 255, 0, 255, 255); // Magenta color
    }

    drawCircle(renderer, x, y, m_genome.size, true);
}

void Organism::eat() {
    foods_eaten++;
    // Increase size
    m_genome.size += 1;
    if (m_genome.size > MAX_ORGANISM_SIZE) {
        m_genome.size = MAX_ORGANISM_SIZE;
    }

    energy_lvl += FOOD_ENERGY;
    if (energy_lvl > max_energy_lvl) {
        energy_lvl = max_energy_lvl;
    }
}

void Organism::setDirection(Direction d) { m_direction = d; }
Direction Organism::getDirection() const { return m_direction; }

Genome Organism::getGenome() const {
    return m_genome;
}

uint32_t Organism::getSector(int width, int height) {
    int sector_x = x / (width / 3);
    int sector_y = y / (height / 3);
    return sector_y * 3 + sector_x;
}