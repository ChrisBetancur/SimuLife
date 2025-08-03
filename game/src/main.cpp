#include <SDL.h>
#include <iostream>

#include <sprites.h>
#include <map.h>
#include <food.h>
#include <organism.h>
#include <wall.h>
#include <stdint.h>
#include <game.h>
#include <logger.h>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

#define MAX_COLORS 8

#include <nn_api.h>
#include <agent.h>

int main(int argc, char* argv[]) {
    Logger::getInstance().init("system.log");
    Game game;
    game.run();
}
