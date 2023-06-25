import copy
import time

import requests

import utils as U

from agents import ActionAgent
from agents import CriticAgent
from agents import CurriculumAgent
from agents import SkillManager
from env import VoyagerEnv


# parse model
import getopt
import sys

openai_model = "gpt-3.5-turbo-0613"
task="Collect clue afqmc train"
context="You can collect clue afqmc train parquet in Chinese LLM Collector."
argv = sys.argv[1:]
try:
    options, args = getopt.getopt(argv, "m:t:c:", ["model =", "task =", "context ="])
except:
    print("Error Message ")

for name, value in options:
    if name in ['-m', '--model']:
        openai_model = value
    elif name in ['-t', '--task']:
        task = value
    elif name in ['-c', '--context']:
        context = value
print("USING:"+openai_model)
print('task:'+task)
print('context:'+context)
recorder = U.EventRecorder()
llm_recorder = U.EventRecorder()

resume = False # if resume the skills
# Agents BEGIN.
action_agent=ActionAgent(model_name=openai_model)
critic_agent=CriticAgent(model_name=openai_model, llm_recorder=llm_recorder)
curriculum_agent = CurriculumAgent(
    model_name=openai_model,
    core_inventory_items=r".*_log|.*_planks|stick|crafting_table|furnace"
        r"|cobblestone|dirt|coal|.*_pickaxe|.*_sword|.*_axe",
    llm_recorder=llm_recorder,
    resume=resume
)
skill_manager = SkillManager(model_name=openai_model, llm_recorder=llm_recorder, resume=resume)
env=VoyagerEnv(server_host='http://transformers.science', server_port=33000)
# Agents END.


#CONSTANT
env_wait_ticks=20

max_iterations=1
action_agent_task_max_retries=2
reset_env=True
reset_placed_if_failed=False

#global var
action_agent_rollout_num_iter = -1
messages = []
conversations = []
last_events = []


def reset(task, context="", reset_env=True):
    global action_agent_rollout_num_iter, messages, conversations, llm_recorder
    action_agent_rollout_num_iter = 0
    # step to peek an observation
    events = env.step(
        ""
    )
    skills = skill_manager.retrieve_skills(query=context)
    print(
        f"\033[33mRender Action Agent system message with {len(skills)} control_primitives\033[0m"
    )
    system_message = action_agent.render_system_message(skills=skills)
    human_message = action_agent.render_human_message(
        events=events, code="", task=task, context=context, critique=""
    )
    messages = [system_message, human_message]
    print(
        f"\033[32m****Action Agent human message****\n{human_message.content}\033[0m"
    )
    assert len(messages) == 2
    conversations = []
    llm_recorder.record([system_message.content, human_message.content], "llm-action")
    return messages

def step(task, context):
    global action_agent_rollout_num_iter, messages, conversations, last_events, llm_recorder, recorder, skill_manager
    info = {}
    info["success"] = False
    if task.startswith("Collect"):
        task = task.replace("Collect", "").strip()
    parts = task.split(" ")
    data = {
        "dataset_name": parts[0]
    }
    res = requests.post(
        f"{env.server}/query_subdb", json=data, timeout=env.request_timeout
    )
    if res.status_code != 200:
        raise RuntimeError("Failed to request CLLMC server")
    returned_data = res.json()
    subsets = returned_data["data"]
    info["task"] = task
    info["dataset_name"] = parts[0]
    info["subsets"] = subsets
    if parts[1] not in subsets:
        return "subset is incorrect", 0, False, info
    data = {
        "dataset_name": parts[0],
        "working_dataset_name": parts[1],
        "split": parts[2]
    }
    info.update(data)
    if len(parts) > 3:
        if len(parts) == 4  and '/' in parts[3]:
            data['current_part'] = parts[3].split('/')[0]
            data['total_part'] = parts[3].split('/')[1]
        if len(parts) > 4:
            data['current_part'] = parts[3]
            data['total_part'] = parts[4]
    res = requests.post(
        f"{env.server}/query_subdb_split", json=data, timeout=env.request_timeout
    )
    if res.status_code != 200:
        raise RuntimeError("Failed to request CLLMC server")
    returned_data = res.json()
    parquet_url = returned_data['data']
    sql = "SELECT count(*) FROM 'local_parquet' LIMIT(10)"
    data = {
        "url": parquet_url,
        "sql": sql
    }
    res = requests.post(
        f"{env.server}/peek_parquet", json=data, timeout=env.request_timeout
    )
    if res.status_code != 200:
        raise RuntimeError("Failed to request CLLMC server")
    returned_data = res.json()
    info.update(returned_data)
    info["success"] = True
    info.update(data)
    return f"Task {task} is finished with success!", 0, True, info


def rollout(task, context, reset_env=True):
    while True:
        messages, reward, done, info = step(task, context)
        if done:
            break
    return messages, reward, done, info


if __name__ == "__main__":
    last_events = env.step("")
    global_info = None
    while True:
        if recorder.iteration > max_iterations:
            print("Iteration limit reached")
            break
        tasks, context = curriculum_agent.propose_next_task(
            cold_task=task,
            cold_context=context,
            events=last_events,
            env_info=global_info,
            max_retries=5,
        )
        llm_recorder.record([tasks, context], "llm-task")
        # solve all the tasks.
        for task in tasks:
            print(
                f"\033[35mStarting task {task} for at most {action_agent_task_max_retries} times\033[0m"
            )
            try:
                messages, reward, done, info = rollout(
                    task=task,
                    context=context,
                    reset_env=reset_env,
                )
            except Exception as e:
                time.sleep(3)  # wait for mineflayer to exit
                info = {
                    "success": False,
                }
                # use red color background to print the error
                print("Your last round rollout terminated due to error:")
                print(f"\033[41m{e}\033[0m")
            if (
                    task == "Place and deposit useless items into a chest"
                    or task.startswith("Deposit useless items into the chest at")
            ):
                continue
            if info["success"]:
                print(f"\033[35mCompleted task {task}.\033[0m")
                skill_manager.add_skill(
                    program_name=task,
                    program_code=info["url"] + "\n\n" + info["sql"],
                )
                curriculum_agent.completed_tasks.append(task)
            else:
                curriculum_agent.failed_tasks.append(task)
                print(
                    f"\033[35mFailed to complete task {task}. Skipping to next task.\033[0m"
                )
            # clean up tasks and dump to disk
            curriculum_agent.clean_up_tasks()
            print(
                f"\033[35mCompleted tasks: {', '.join(curriculum_agent.completed_tasks)}\033[0m"
            )
            print(
                f"\033[35mFailed tasks: {', '.join(curriculum_agent.failed_tasks)}\033[0m"
            )
            global_info = info
        last_events = env.step("")
        # GPT-3.5 sleep time.
        time.sleep(env_wait_ticks)
        recorder.record({
            "event": last_events,
            "info": global_info
        }, context)
