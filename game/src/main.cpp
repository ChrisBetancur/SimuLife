#include <SDL.h>
#include <iostream>

#include <sprites.h>
#include <map.h>
#include <food.h>
#include <organism.h>
#include <wall.h>
#include <stdint.h>
#include <game.h>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

#define MAX_COLORS 8

#include <nn_api.h>
#include <agent.h>

int main(int argc, char* argv[]) {
    Game game;
    game.run();
}
