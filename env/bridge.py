import os.path
import time
import warnings
from typing import SupportsFloat, Any, Tuple, Dict

import requests
import json

import gymnasium as gym
from gymnasium.core import ObsType

import utils as U


class VoyagerEnv(gym.Env):
    def __init__(
        self,
        server_host="http://127.0.0.1",
        server_port=3000,
        request_timeout=600,
        log_path="./logs",
    ):
        self.server = f"{server_host}:{server_port}"
        self.server_port = server_port
        self.request_timeout = request_timeout
        self.log_path = log_path
        # self.mineflayer = self.get_mineflayer_process(server_port)
        self.has_reset = False
        self.reset_options = None
        self.connected = False
        self.server_paused = False

    # def get_mineflayer_process(self, server_port):
    #     U.f_mkdir(self.log_path, "mineflayer")
    #     # file_path = os.path.abspath(os.path.dirname(__file__))
    #     file_path = os.sep.join(os.path.abspath(os.path.dirname(__file__)).split(os.sep)[:-2])
    #     return SubprocessMonitor(
    #         commands=[
    #             "node",
    #             U.f_join(file_path, "Open-Node-Research/index.js"),
    #             str(server_port),
    #         ],
    #         name="mineflayer",
    #         ready_match=r"Server started on port (\d+)",
    #         log_path=U.f_join(self.log_path, "mineflayer"),
    #     )

    # def get_mc_instance(self):
    #     print("Creating Minecraft server")
    #     U.f_mkdir(self.log_path, "minecraft")
    #     return MinecraftInstance(
    #         # **self.azure_login,
    #         mineflayer=self.mineflayer,
    #         log_path=U.f_join(self.log_path, "minecraft"),
    #     )

    def check_process(self):
        if self.mc_instance and not self.mc_instance.is_running:
            # if self.mc_instance:
            #     self.mc_instance.check_process()
            #     if not self.mc_instance.is_running:
            print("Starting Minecraft server")
            self.mc_instance.run()
            self.mc_port = self.mc_instance.port
            self.reset_options["port"] = self.mc_instance.port
            print(f"Server started on port {self.reset_options['port']}")
        retry = 0
        while not self.mineflayer.is_running:
            print("Mineflayer process has exited, restarting")
            self.mineflayer.run()
            if not self.mineflayer.is_running:
                if retry > 3:
                    raise RuntimeError("Mineflayer process failed to start")
                else:
                    continue
            print(self.mineflayer.ready_line)
            res = requests.post(
                f"{self.server}/start",
                json=self.reset_options,
                timeout=self.request_timeout,
            )
            if res.status_code != 200:
                self.mineflayer.stop()
                raise RuntimeError(
                    f"Minecraft server reply with code {res.status_code}\n err {res.reason}"
                )
            return res.json()

    def step(
        self,
        code: str,
        programs: str = "",
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        # if not self.has_reset:
        #     raise RuntimeError("Environment has not been reset yet")
        # self.check_process()
        # self.unpause()
        data = {
            "url": code,
            "sql": programs,
        }
        res = requests.post(
            f"{self.server}/step", json=data, timeout=self.request_timeout
        )
        if res.status_code != 200:
            raise RuntimeError("Failed to step Minecraft server")
        returned_data = res.json()
        # self.pause()
        return returned_data["status"]

    def render(self):
        raise NotImplementedError("render is not implemented")

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        if options is None:
            options = {}

        if options.get("inventory", {}) and options.get("mode", "hard") != "hard":
            raise RuntimeError("inventory can only be set when options is hard")

        self.reset_options = {
            "server": "127.0.0.1",
            "port": self.mc_port,
            "reset": options.get("mode", "hard"),
            "inventory": options.get("inventory", {}),
            "equipment": options.get("equipment", []),
            "spread": options.get("spread", False),
            "waitTicks": options.get("wait_ticks", 5),
            "position": options.get("position", None),
        }

        self.unpause()
        self.mineflayer.stop()
        time.sleep(1)  # wait for mineflayer to exit

        returned_data = self.check_process()
        self.has_reset = True
        self.connected = True
        # All the reset in step will be soft
        self.reset_options["reset"] = "soft"
        self.pause()
        return json.loads(returned_data)

    def close(self):
        self.unpause()
        if self.connected:
            res = requests.post(f"{self.server}/stop")
            if res.status_code == 200:
                self.connected = False
        if self.mc_instance:
            self.mc_instance.stop()
        self.mineflayer.stop()
        return not self.connected

    def pause(self):
        if self.mineflayer.is_running and not self.server_paused:
            res = requests.post(f"{self.server}/pause")
            if res.status_code == 200:
                self.server_paused = True
        return self.server_paused

    def unpause(self):
        if self.mineflayer.is_running and self.server_paused:
            res = requests.post(f"{self.server}/pause")
            if res.status_code == 200:
                self.server_paused = False
            else:
                print(res.json())
        return self.server_paused
