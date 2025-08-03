#ifndef GAME_H
#define GAME_H

#include <SDL.h>
#include <SDL_ttf.h>
#include <iostream>
#include <sprites.h>
#include <map.h>
#include <food.h>
#include <organism.h>
#include <wall.h>
#include <stdint.h>
#include <nn_api.h>
#include <agent.h>
#include <stdbool.h>


class Game {
    private:
        SDL_Window* m_window;
        SDL_Renderer* m_renderer;
        Map* m_map;
        Organism* m_organism;
        Agent* m_agent;
        Trainer* m_trainer;
        
        enum class GameState { MENU, RUNNING, QUIT };
        GameState m_currentState;
        
        int m_totalEpisodes;
        int m_currentEpisode;

        bool m_rndEnabled;

        int timestep = 0;
        std::vector<std::string> m_policies; // List of policies to choose from
        std::vector<bool> m_selectedPolicies; // Track selected policies
        
    public:
        Game();
        ~Game();
        void run();
        void showMenu();
        void runEpisodes(int episodes);
};


#endif