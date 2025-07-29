#include <game.h>



Game::Game() : m_currentState(State::MENU), m_totalEpisodes(0), m_currentEpisode(0) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL init failed: " << SDL_GetError() << std::endl;
        exit(1);
    }

    // Initialize TTF subsystem
    if (TTF_Init() < 0) {
        std::cerr << "TTF init failed: " << TTF_GetError() << std::endl;
        exit(1);
    }

    m_window = SDL_CreateWindow("SimuLife", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                             800, 600, SDL_WINDOW_SHOWN);
    m_renderer = SDL_CreateRenderer(m_window, -1, SDL_RENDERER_ACCELERATED);
    
    m_map = new Map(800, 600);

    std::random_device rd;
    std::mt19937 gen(rd());

    // Center at 400px, stddev ≈ 800/6 ≈ 133px covers ±2σ ≈ 66%
    // Use smaller σ (≈800/9≈89px) to tighten to ≈90% within ±3σ (~±267px)
    std::normal_distribution<> distX(400.0, 89.0);
    std::normal_distribution<> distY(300.0, 67.0);  // for 600 height

    int x = std::clamp<int>(std::round(distX(gen)), 10, 790);
    int y = std::clamp<int>(std::round(distY(gen)), 10, 590);

        
    m_organism = new Organism(x, y, {1, MAX_ORGANISM_VISION_DEPTH, MAX_ORGANISM_SPEED, 10});
    //m_map->addOrganism(x, y, {1, MAX_ORGANISM_VISION_DEPTH, MAX_ORGANISM_SPEED, MIN_ORGANISM_SIZE});

    m_agent = new Agent(m_organism);
    m_trainer = new Trainer(m_agent, m_map, 0.9, 0.001, "test_model");
}

Game::~Game() {
    delete m_map;
    delete m_organism;
    delete m_agent;
    delete m_trainer;

    SDL_DestroyRenderer(m_renderer);
    SDL_DestroyWindow(m_window);
    SDL_Quit();
}

void drawText(SDL_Renderer* renderer,
    const std::string& text,
    int x, int y,
    SDL_Color color)
{
static TTF_Font* font = nullptr;
    if (!font) {
    font = TTF_OpenFont("assets/ChicagoFLF.ttf", 24);
    if (!font) {
    std::cerr << "Failed to open font: " << TTF_GetError() << std::endl;
    return;
    }
    }

    SDL_Surface* surface = TTF_RenderText_Solid(font, text.c_str(), color);
    if (!surface) {
    std::cerr << "TTF_RenderText_Solid error: " << TTF_GetError() << std::endl;
    return;
    }

    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (!texture) {
    std::cerr << "SDL_CreateTextureFromSurface error: " << SDL_GetError() << std::endl;
    SDL_FreeSurface(surface);
    return;
    }

    SDL_Rect dest{ x, y, surface->w, surface->h };
    SDL_RenderCopy(renderer, texture, nullptr, &dest);

    SDL_FreeSurface(surface);
    SDL_DestroyTexture(texture);
}

void Game::run() {

    bool running = true;

    while (running) {
        switch (m_currentState) {
            case State::MENU:
                showMenu();
                break;
            case State::RUNNING:
                runEpisodes(m_totalEpisodes);
                break;

            case State::QUIT:
                running = false;
                break;
        }
    }
}

void Game::runEpisodes(int episodes) {
    for (int i = 0; i < episodes; ++i) {
        m_map->reset();
        std::random_device rd;
        std::mt19937 gen(rd());
    
        // Center at 400px, stddev ≈ 800/6 ≈ 133px covers ±2σ ≈ 66%
        // Use smaller σ (≈800/9≈89px) to tighten to ≈90% within ±3σ (~±267px)
        std::normal_distribution<> distX(400.0, 89.0);
        std::normal_distribution<> distY(300.0, 67.0);  // for 600 height
    
        int x = std::clamp<int>(std::round(distX(gen)), 10, 790);
        int y = std::clamp<int>(std::round(distY(gen)), 10, 590);
        
        m_organism->reset(x, y);

        bool running = true;
        SDL_Event event;

        const uint8_t* keyboardState = SDL_GetKeyboardState(nullptr);

        SDL_SetRenderDrawColor(m_renderer, 255, 255, 255, 255);
        SDL_RenderClear(m_renderer);
        
        while (running) {
            // Process events
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    running = false;
                }
            }

            // Clear the screen EVERY FRAME
            SDL_SetRenderDrawColor(m_renderer, 255, 255, 255, 255);
            SDL_RenderClear(m_renderer);

            int x, y;
            m_organism->getPosition(x, y);

            int dx = 0, dy = 0;

            // Get the action from the agent
            //agent->updateState(gameMap);
            Action action = m_agent->chooseAction();
            switch (action.direction) {
                case UP:    dy = -1; break;
                case DOWN:  dy = +1; break;
                case LEFT:  dx = -1; break;
                case RIGHT: dx = +1; break;
            }

            if (dx != 0 || dy != 0) {
                // proposed new position
                int speed = m_organism->getGenome().speed;
                int newX = x + dx * speed;
                int newY = y + dy * speed;

                timestep++;

                std::vector<double> food_rates = m_map->getFoodCounts();

                // compute rates food_counts/timestep
                for (int i = 0; i < food_rates.size(); ++i) {
                    food_rates[i] = static_cast<double>(food_rates[i] / (timestep + 1));
                }

                uint32_t sector = m_organism->getSector(m_map->getWidth(), m_map->getHeight());
            
                // only move if there is no wall at the target
                if (!m_map->isWall(newX, newY)) {
                    // print check
                    float reward = computeReward(m_agent->getState(), action, food_rates);
                    // passed reward print check

                    running = m_organism->move(dx, dy);
                    m_agent->updateState(m_map);
                    m_trainer->learn(m_agent->getState(), action, reward); // reward is 0 for now
                }
                else {
                    // print check
                    float reward = computeReward(m_agent->getState(), action, food_rates);
                    // passed reward print check

                    running = m_organism->move(0, 0);
                    m_agent->updateState(m_map);
                    m_trainer->learn(m_agent->getState(), action, reward); // reward is 0 for now
                }
            }

            int org_x, org_y;
            m_organism->getPosition(org_x, org_y);

            m_map->organismCollisionFood((Organism*) m_organism);

            m_map->draw_map(m_renderer);
            m_organism->draw(m_renderer);

            
            SDL_RenderPresent(m_renderer);
            SDL_Delay(10);
        }

        save_nn_model(0, 0, "test_model");
        // svae rnd predictor model
        save_nn_model(0, 2, "rnd_models/test_model/predictor");
        // save rnd target model
        save_nn_model(0, 3, "rnd_models/test_model/target");

        SDL_SetRenderDrawColor(m_renderer, 255, 255, 255, 255);
        SDL_RenderClear(m_renderer);
        SDL_RenderPresent(m_renderer);

        SDL_Delay(500);

        timestep = 0;
    }

    m_currentState = State::MENU;
}

void Game::showMenu() {
    bool inMenu = true;
    std::string episodeInput;
    SDL_StartTextInput();

    while (inMenu) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    inMenu = false;
                    m_currentState = State::QUIT;
                    return;
                    
                case SDL_MOUSEBUTTONDOWN:
                    if (event.button.button == SDL_BUTTON_LEFT) {
                        int x = event.button.x;
                        int y = event.button.y;
                        
                        // Check if click is within "Start" button area
                        if (x > 300 && x < 500 && y > 400 && y < 450) {
                            try {
                                m_totalEpisodes = std::stoi(episodeInput);
                                if (m_totalEpisodes > 0) {
                                    inMenu = false;
                                    m_currentState = State::RUNNING;
                                }
                            } catch (...) {
                                // Invalid input
                            }
                        }
                    }
                    break;
                    
                case SDL_TEXTINPUT:
                    episodeInput += event.text.text;
                    break;
                    
                case SDL_KEYDOWN:
                    if (event.key.keysym.sym == SDLK_BACKSPACE && !episodeInput.empty()) {
                        episodeInput.pop_back();
                    }
                    break;
            }
        }

        // Draw menu
        SDL_SetRenderDrawColor(m_renderer, 50, 50, 50, 255);
        SDL_RenderClear(m_renderer);

        // Draw title
        drawText(m_renderer, "Simulife AI Training", 260, 100, {255, 255, 255});
        
        // Draw input box
        SDL_Rect inputRect = {200, 300, 400, 40};
        SDL_SetRenderDrawColor(m_renderer, 255, 255, 255, 255);
        SDL_RenderFillRect(m_renderer, &inputRect);
        drawText(m_renderer, "Episodes: " + episodeInput, 210, 305, {0, 0, 0});
        
        // Draw start button
        SDL_Rect buttonRect = {300, 400, 200, 50};
        SDL_SetRenderDrawColor(m_renderer, 0, 255, 0, 255);
        SDL_RenderFillRect(m_renderer, &buttonRect);
        drawText(m_renderer, "Start Training", 312, 410, {0, 0, 0});

        SDL_RenderPresent(m_renderer);
    }
    SDL_StopTextInput();
}