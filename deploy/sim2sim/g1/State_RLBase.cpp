#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <filesystem>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        cmd = key_commands[key];
    }
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    std::filesystem::path onnx_path = policy_dir / "exported" / "policy.onnx";
    if (cfg["onnx_path"]) {
        onnx_path = cfg["onnx_path"].as<std::string>();
        if (onnx_path.is_relative()) {
            onnx_path = param::proj_dir / onnx_path;
        }
    }

    auto deploy_config_path = cfg["deploy_config"]
        ? std::filesystem::path(cfg["deploy_config"].as<std::string>())
        : policy_dir / "params" / "deploy.yaml";
    if (deploy_config_path.is_relative()) {
        deploy_config_path = param::proj_dir / deploy_config_path;
    }
    if (!std::filesystem::exists(deploy_config_path)) {
        throw std::runtime_error("Deploy config does not exist: " + deploy_config_path.string());
    }
    spdlog::info("Deploy config: {}", deploy_config_path.string());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(deploy_config_path),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(onnx_path.string());

    const std::string diagnostics_path =
        env->cfg["diagnostics_csv"].as<std::string>("");
    if (!diagnostics_path.empty()) {
        const std::filesystem::path path(diagnostics_path);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        diagnostics_csv_.open(path, std::ios::out | std::ios::trunc);
        if (!diagnostics_csv_) {
            throw std::runtime_error("Could not open diagnostics CSV: " + diagnostics_path);
        }

        diagnostics_csv_ << std::setprecision(9);
        spdlog::info("Getup diagnostics CSV enabled: {}", path.string());
    }

    // Do not copy the locomotion bad-orientation guard here: a fallen pose is
    // the intended initial condition for Getup. The joystick FSM transition
    // remains the operator's explicit stop path.
}

void State_RLBase::write_diagnostics(double elapsed_s, double policy_step_ms)
{
    const auto obs_it = env->last_observations.find("obs");
    if (obs_it == env->last_observations.end()) {
        throw std::runtime_error("Diagnostics expected the flat observation group named 'obs'.");
    }

    const auto action = env->action_manager->action();
    const auto q_target = env->action_manager->processed_actions();
    const auto& data = env->robot->data;
    if (!diagnostics_header_written_) {
        const auto write_headers = [this](const std::string& prefix, size_t count) {
            for (size_t i = 0; i < count; ++i) diagnostics_csv_ << ',' << prefix << i;
        };
        diagnostics_csv_ << "step,time_s,policy_step_ms";
        write_headers("obs_", obs_it->second.size());
        write_headers("action_", action.size());
        write_headers("q_target_", q_target.size());
        write_headers("q_", data.joint_pos.size());
        write_headers("dq_", data.joint_vel.size());
        write_headers("tau_est_", data.joint_torque_est.size());
        write_headers("q_target_error_", std::min<size_t>(q_target.size(), data.joint_pos.size()));
        diagnostics_csv_ << '\n';
        diagnostics_header_written_ = true;
    }
    if (q_target.size() != data.joint_pos.size()) {
        throw std::runtime_error("Diagnostics target and joint position lengths differ.");
    }
    const auto write_vector = [this](const auto& values) {
        for (int i = 0; i < values.size(); ++i) diagnostics_csv_ << ',' << values[i];
    };

    diagnostics_csv_ << diagnostics_step_++ << ',' << elapsed_s << ',' << policy_step_ms;
    write_vector(obs_it->second);
    write_vector(action);
    write_vector(q_target);
    write_vector(data.joint_pos);
    write_vector(data.joint_vel);
    write_vector(data.joint_torque_est);
    for (int i = 0; i < q_target.size(); ++i) {
        diagnostics_csv_ << ',' << (q_target[i] - data.joint_pos[i]);
    }
    diagnostics_csv_ << '\n';
    if (diagnostics_step_ % 10 == 0) diagnostics_csv_.flush();
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    const auto& joint_ids = env->robot->data.joint_ids_map;
    for(size_t i = 0; i < joint_ids.size(); i++) {
        lowcmd->msg_.motor_cmd()[joint_ids[i]].q() = action[i];
    }
}
