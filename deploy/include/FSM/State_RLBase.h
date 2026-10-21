// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include <chrono>
#include <fstream>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        for (size_t i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            const auto motor_id = env->robot->data.joint_ids_map[i];
            lowcmd->msg_.motor_cmd()[motor_id].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[motor_id].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[motor_id].dq() = 0;
            lowcmd->msg_.motor_cmd()[motor_id].tau() = 0;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            using steady_clock = std::chrono::steady_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            const auto diagnostics_start = steady_clock::now();
            env->reset();

            while (policy_thread_running)
            {
                const auto step_start = steady_clock::now();
                env->step();
                const auto step_end = steady_clock::now();
                if (diagnostics_csv_.is_open()) {
                    const double elapsed_s = std::chrono::duration<double>(
                        step_end - diagnostics_start).count();
                    const double policy_step_ms = std::chrono::duration<double, std::milli>(
                        step_end - step_start).count();
                    write_diagnostics(elapsed_s, policy_step_ms);
                }

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();

    void write_diagnostics(double elapsed_s, double policy_step_ms);
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
        if (diagnostics_csv_.is_open()) {
            diagnostics_csv_.flush();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::ofstream diagnostics_csv_;
    long diagnostics_step_ = 0;
    bool diagnostics_header_written_ = false;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)
