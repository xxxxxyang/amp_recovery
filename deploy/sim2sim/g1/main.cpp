#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <thread>
#include <cctype>
#include <stdexcept>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

constexpr uint16_t FSM_UI_PORT = 9876;

void start_fsm_ui_command_server()
{
    const int socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd < 0) {
        spdlog::error("FSM UI: failed to create UDP socket");
        return;
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(FSM_UI_PORT);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(socket_fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0) {
        spdlog::error("FSM UI: cannot bind 127.0.0.1:{}", FSM_UI_PORT);
        close(socket_fd);
        return;
    }
    spdlog::info("FSM UI command listener: 127.0.0.1:{}", FSM_UI_PORT);

    while (true) {
        char buffer[128]{};
        sockaddr_in client{};
        socklen_t client_len = sizeof(client);
        const auto count = recvfrom(socket_fd, buffer, sizeof(buffer) - 1, 0,
                                    reinterpret_cast<sockaddr *>(&client), &client_len);
        if (count <= 0) continue;

        std::string state_name(buffer, static_cast<size_t>(count));
        while (!state_name.empty() && std::isspace(static_cast<unsigned char>(state_name.back()))) {
            state_name.pop_back();
        }
        const auto state = FSMStringMap.right.find(state_name);
        std::string response;
        if (state == FSMStringMap.right.end()) {
            response = "error: unknown FSM state";
        } else {
            FSMState::requested_state.store(state->second, std::memory_order_release);
            response = "queued:" + state_name;
        }
        sendto(socket_fd, response.data(), response.size(), 0,
               reinterpret_cast<sockaddr *>(&client), client_len);
    }
}

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);
    if (vm.count("onnx")) {
        param::config["FSM"]["Getup"]["onnx_path"] = vm["onnx"].as<std::string>();
    }
    if (vm.count("deploy-config")) {
        param::config["FSM"]["Getup"]["deploy_config"] = vm["deploy-config"].as<std::string>();
    }

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof sim2sim Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();
    std::thread(start_fsm_ui_command_server).detach();

    std::cout << "Start unitree_mujoco with a G1 scene in a fallen pose.\n";
    std::cout << "Gamepad: [L2 + Y] PD_STAND, [L2 + Up] Getup, [L2 + B] Passive.\n";
    std::cout << "Optional FSM UI: python3 deploy/sim2sim/g1/fsm_ui.py\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}
