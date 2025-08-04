#include <game.h>
#include <rl_utils.h>
#include <logger.h>



Game::Game() : m_currentState(GameState::MENU), m_totalEpisodes(0), m_currentEpisode(0) {
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
            case GameState::MENU:
                showMenu();
                break;
            case GameState::RUNNING:
                runEpisodes(m_totalEpisodes);
                break;

            case GameState::QUIT:
                running = false;
                break;
        }
    }
}



void Game::runEpisodes(int episodes) {
    bool policySelected = false;

    std::cout << "Available policies:" << m_policies.size()<< std::endl;
    for (int i = 0; i < m_policies.size(); ++i) {
        // print check
        std::cout << "Policy: " << m_policies[i] << ", Selected: " << (m_selectedPolicies[i] ? "Yes" : "No") << std::endl;
        if (m_selectedPolicies[i]) {
            if (m_policies[i] == "Epsilon-Greedy") {
                // printcheck
                std::cout << "Using EpsilonGreedy policy" << std::endl;
                // POLICY TYPE IS A ENUM IN rl_utils.h
                m_agent->setPolicy(PolicyType::EPSILON_GREEDY);
                policySelected = true;
            } else if (m_policies[i] == "Boltzmann") {
                // printcheck
                std::cout << "Using Boltzmann policy" << std::endl;
                m_agent->setPolicy(PolicyType::BOLTZMANN);
                policySelected = true;
            }
        }
    }

    if (!policySelected) {
        std::cerr << "No policy selected. Please select at least one policy." << std::endl;
        m_currentState = GameState::MENU;
        return;
    }
    
    for (int i = 0; i < episodes; ++i) {

        Logger::getInstance().log(LogType::INFO, "---------- Episode " + std::to_string(i + 1) + " of " + std::to_string(episodes) + " ----------");
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
            m_agent->updateState(m_map);
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
                    double reward = computeReward(m_agent->getState(), action, food_rates, sector, m_rndEnabled);
                    // passed reward print check

                    running = m_organism->move(dx, dy);
                    m_agent->updateState(m_map);
                    m_trainer->learn(m_agent->getState(), action, reward); // reward is 0 for now
                }
                else {
                    // print check
                    double reward = computeReward(m_agent->getState(), action, food_rates, sector, m_rndEnabled);
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
            m_map->drawVision(m_renderer);

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

        Logger::getInstance().log(LogType::DEBUG, "Final Timestep: " + std::to_string(timestep));

        timestep = 0;
        Logger::getInstance().log(LogType::DEBUG, "-------- End of Episode " + std::to_string(i + 1) + " --------\n\n");

    }

    m_currentState = GameState::MENU;
}


void Game::showMenu() {
    bool inMenu = true;
    std::string episodeInput;
    SDL_StartTextInput();

    // Policy options and RND toggle state
    m_policies = { "Epsilon-Greedy", "Boltzmann" };
    std::vector<bool> policySelected(m_policies.size(), false);

    bool rndEnabled = false;

    int inputX = 50, inputY = 200;

    // UI layout constants
    const int menuWidth = 800, menuHeight = 600;
    const SDL_Rect inputRect   = {inputX, inputY, 400, 40};
    const SDL_Rect startButton = {300, 500, 200, 50};
    const SDL_Color bgColor    = {50, 50, 50, 255};
    const SDL_Color textColor  = {255, 255, 255, 255};
    const SDL_Color boxColor   = {200, 200, 200, 255};

    int rndSwitchX = 50, rndSwitchY = 400;
    int policyBoxX = 50, policyBoxY = 265;

    while (inMenu) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    inMenu = false;
                    m_currentState = GameState::QUIT;
                    break;
                case SDL_MOUSEBUTTONDOWN: {
                    int mx = event.button.x;
                    int my = event.button.y;

                    bool hasPolicySelected = false;

                    for (bool policy : policySelected) {
                        if (policy) {
                            hasPolicySelected = true;
                            break;
                        }
                    }

                    // Toggle checkpoints for each policy
                    for (size_t i = 0; i < m_policies.size(); ++i) {
                        SDL_Rect chk = {policyBoxX, policyBoxY + int(i) * 40, 20, 20};
                        if (mx >= chk.x && mx <= chk.x + chk.w && my >= chk.y && my <= chk.y + chk.h) {
                            if (!hasPolicySelected) {
                                hasPolicySelected = true;
                                policySelected[i] = !policySelected[i];
                            }
                            else {
                                // If already selected, toggle off
                                if (policySelected[i]) {
                                    policySelected[i] = false;
                                    break; // Skip to next iteration
                                }
                            }
                            
                        }
                    }
                    // Toggle RND switch
                    SDL_Rect rndSwitch = {rndSwitchX, rndSwitchY, 60, 30};
                    if (mx >= rndSwitch.x && mx <= rndSwitch.x + rndSwitch.w && my >= rndSwitch.y && my <= rndSwitch.y + rndSwitch.h) {
                        rndEnabled = !rndEnabled;
                    }
                    // Start button
                    if (mx > startButton.x && mx < startButton.x + startButton.w && my > startButton.y && my < startButton.y + startButton.h) {
                        try {
                            m_totalEpisodes = std::stoi(episodeInput);
                            if (m_totalEpisodes > 0) {
                                inMenu = false;
                                m_currentState = GameState::RUNNING;
                            }
                        } catch (...) {}
                    }
                } break;
                case SDL_TEXTINPUT:
                    episodeInput += event.text.text;
                    break;
                case SDL_KEYDOWN:
                    if (event.key.keysym.sym == SDLK_BACKSPACE && !episodeInput.empty())
                        episodeInput.pop_back();
                    break;
            }
        }

        // Draw background
        SDL_SetRenderDrawColor(m_renderer, bgColor.r, bgColor.g, bgColor.b, bgColor.a);
        SDL_RenderClear(m_renderer);

        // Draw title
        drawText(m_renderer, "SimuLife AI Training", 260, 100, textColor);

        // Draw episode input
        SDL_SetRenderDrawColor(m_renderer, 255, 255, 255, 255);
        SDL_RenderFillRect(m_renderer, &inputRect);
        drawText(m_renderer, "Episodes: " + episodeInput, inputX + 10, inputY + 5, {0,0,0,255});

        // Draw policy checkboxes
        for (size_t i = 0; i < m_policies.size(); ++i) {
            SDL_Rect box = {policyBoxX, policyBoxY + int(i) * 40, 20, 20};
            SDL_SetRenderDrawColor(m_renderer, boxColor.r, boxColor.g, boxColor.b, boxColor.a);
            SDL_RenderFillRect(m_renderer, &box);
            if (policySelected[i]) {
                // draw inner filled box when selected
                SDL_Rect inner = {box.x + 4, box.y + 4, box.w - 8, box.h - 8};
                SDL_SetRenderDrawColor(m_renderer, 0, policyBoxY, 0, 255);
                SDL_RenderFillRect(m_renderer, &inner);
            }
            drawText(m_renderer, m_policies[i], box.x + 30, box.y - 2, textColor);
        }

        // Draw RND toggle switch background
        SDL_Rect rndBg = {rndSwitchX, rndSwitchY, 60, 30};
        SDL_SetRenderDrawColor(m_renderer, boxColor.r, boxColor.g, boxColor.b, boxColor.a);
        SDL_RenderFillRect(m_renderer, &rndBg);
        // Draw switch handle
        SDL_Rect handle = rndEnabled
            ? SDL_Rect{rndSwitchX + rndBg.w - 28, rndSwitchY + 2, 26, 26}
            : SDL_Rect{rndSwitchX + 2,         rndSwitchY + 2, 26, 26};
        SDL_SetRenderDrawColor(m_renderer, rndEnabled ? 0 : rndSwitchY - 50, rndEnabled ? rndSwitchY : 100, 0, 255);
        SDL_RenderFillRect(m_renderer, &handle);
        drawText(m_renderer, std::string("RND: ") + (rndEnabled ? "ON" : "OFF"), rndSwitchX, rndSwitchY + 40, textColor);

        // Draw Start button
        SDL_SetRenderDrawColor(m_renderer, 0, 255, 0, 255);
        SDL_RenderFillRect(m_renderer, &startButton);
        drawText(m_renderer, "Start Training", startButton.x + 12, startButton.y + 12, {0,0,0,255});

        SDL_RenderPresent(m_renderer);
        SDL_Delay(10);
    }
    SDL_StopTextInput();
    // Store selected policies and rndEnabled in member variables, if needed
    m_selectedPolicies = policySelected;
    m_rndEnabled = rndEnabled;
}
