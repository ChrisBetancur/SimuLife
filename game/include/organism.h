#ifndef ORGANISM_H
#define ORGANISM_H
#include <SDL.h>
#include <iostream>
#include <sprites.h>

#define MAX_ORGANISM_SIZE 50
#define MIN_ORGANISM_SIZE 5
#define MAX_ORGANISM_SPEED 5
#define MIN_ORGANISM_SPEED 1
#define MAX_ORGANISM_VISION_DEPTH 5
#define MIN_ORGANISM_VISION_DEPTH 1
#define FOOD_ENERGY 10

struct Genome {
    uint32_t gender;
    uint32_t vision_depth;
    uint32_t speed;
    uint32_t size;
};

class Organism : public Sprite {
    private:
        Direction m_direction; // Direction of the organism
        float max_energy_lvl; // Maximum energy level of the organism
        float energy_lvl; // Energy level of the organism

        const float basal_rate;  // energy/sec just to stay alive
        const float speed_coeff; // extra energy/sec per unit speed
        const float starve_thresh; // fraction of maxEnergy below which you fatigue
        const float starve_accel; // extra multiplier when starving
        const uint32_t original_size; // original size of the organism

        Genome m_genome; // Genome of the organism

        enum Gender {
            MALE = 0,
            FEMALE = 1
        };

        // Organism properties
        uint32_t foods_eaten = 0;

    public:
        Organism(int x, int y, Genome genome);

        void reset(int x, int y);

        void applyEnergyCost();
        
        bool move(int dx, int dy);

        void draw(SDL_Renderer* renderer) override;

        void eat();

        Genome getGenome() const;

        void setDirection(Direction d);

        Direction getDirection() const;

        uint32_t getEnergy() const { return energy_lvl; }

        uint32_t getSector(int width, int height);

        uint32_t foodCount() const { return foods_eaten; }

};

#endif