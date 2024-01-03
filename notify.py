#!/usr/bin/env python3

import numba.cuda as cuda
from psutil import Process
import json
import telegram
from telegram.ext import Updater, CommandHandler
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
import pynvml
import time
from traceback import format_exc


def pbar(current, maximum, size):
    output = "`|"
    sep = round(current / maximum * size)
    for _ in range(sep + 1):
        output += "#"
    for _ in range(sep + 1, size):
        output += " "
    output += "|`"
    return output


def get_user_info(handle):
    procs = [
        Process(proc.pid)
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    ]
    return [(proc.username(), " ".join(proc.cmdline())) for proc in procs]


def get_usage_msg(gpu_id, handle):
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_gb = info.used / (1024**3)
    total_gb = info.total / (1024**3)
    msg = f"GPU {gpu_id} usage is {used_gb:.0f} / {total_gb:.0f} GB\n{pbar(info.used, info.total, 24)}\n"
    uinfos = get_user_info(handle)
    for name, cmdline in uinfos:
        msg += f"😈 {name}: `{cmdline}`\n"
    if not uinfos:
        msg += f"👻 Type `/occ {gpu_id}` to occupy the GPU\n"
    return msg


class NotifyBot:
    def __init__(self):
        super().__init__()

        try:
            with open("config.json", "r") as config_file:
                config = json.load(config_file)
                token = config["token"]
                self._whitelist = config["whitelist"]
        except FileNotFoundError:
            print("You need to have a config.json file in this directory")
            exit(1)

        self._last_thresh = -1

        self.cuda_buffers = {}
        self._updater = Updater(token, use_context=True)
        dp = self._updater.dispatcher
        dp.add_handler(CommandHandler("start", self._register))
        dp.add_handler(CommandHandler("gpu", self._get_gpu))
        dp.add_handler(CommandHandler("g", self._get_gpu_quick))
        dp.add_handler(CommandHandler("occ", self._occupy, pass_args=True))
        dp.add_handler(CommandHandler("rel", self._release, pass_args=True))
        dp.add_handler(CommandHandler("echo", self._echo, pass_args=True))
        dp.add_handler(
            CommandHandler("set_interval", self._set_interval, pass_args=True)
        )
        self._updater.start_polling()

        self.interval = 5
        self._poll_gpu()

    def _register(self, update: Update, context: CallbackContext):
        user_id = update.message.from_user.id
        if user_id not in self._whitelist:
            update.message.reply_text(
                "You are not yet on the whitelist. "
                + f"Add {user_id} to your config to receive notifications from me"
            )
        else:
            update.message.reply_text(
                "Hi! I will notify you when someone starts "
                + "to use the GPU and when it's available again"
            )

    def _echo(self, update: Update, context: CallbackContext):
        print(update.message.from_user.username, "requested echo")
        if len(context.args) != 1:
            print("Invalid usage of /echo")
            update.message.reply_text("Usage: /echo <anything>")
        else:
            update.message.reply_text(context.args[0])

    # ! Restricted
    def _occupy(self, update: Update, context: CallbackContext):
        user_id = update.message.from_user.id
        if user_id not in self._whitelist:
            update.message.reply_text(
                "You are not yet on the whitelist. "
                + f"Add {user_id} to your config to receive notifications from me"
            )
            return

        print(update.message.from_user.username, "tries to occupy some resources")
        if len(context.args) != 1:
            print("Invalid usage of /occ")
            update.message.reply_text("Usage: /occ <gpu_id>")
            return
        else:
            gid = int(context.args[0])
            if gid < 0 or gid >= pynvml.nvmlDeviceGetCount():
                update.message.reply_text(
                    f"Invalid GPU ID. Must be between 0 and {pynvml.nvmlDeviceGetCount() - 1}"
                )
                return

        to_alloc = pynvml.nvmlDeviceGetMemoryInfo(
            pynvml.nvmlDeviceGetHandleByIndex(gid)
        ).free
        to_alloc = int(to_alloc * 0.95)
        cuda.select_device(gid)
        for _ in range(3):
            try:
                self.cuda_buffers[gid] = cuda.device_array((to_alloc,), dtype="byte")
                update.message.reply_text(
                    f"Allocated {to_alloc / (1024**3):f.1}GB on GPU {gid}"
                )
                break
            except cuda.cudadrv.driver.CudaAPIError:
                update.message.reply_text(
                    f"Could not allocate {to_alloc / (1024**3):f.1}GB on GPU {gid}. Trying again with 80% of the memory"
                )
                to_alloc = int(to_alloc * 0.8)

    # ! Restricted
    def _set_interval(self, update: Update, context: CallbackContext):
        user_id = update.message.from_user.id
        if user_id not in self._whitelist:
            update.message.reply_text(
                "You are not yet on the whitelist. "
                + f"Add {user_id} to your config to receive notifications from me"
            )
            return

        print(update.message.from_user.username, "tries to set the interval")
        if len(context.args) != 1:
            print("Invalid usage of /rel")
            update.message.reply_text(
                "Usage: /rel <gpu_id> to release <gpu_id> or simply /rel to release all"
            )
            return

        new_interval = int(context.args[0])
        if new_interval < 1 or new_interval > 60 * 60:
            update.message.reply_text("Interval must be between 1s and 3600s")
            return

        self.interval = new_interval
        update.message.reply_text(f"Interval set to {new_interval}s")

    # ! Restricted
    def _release(self, update: Update, context: CallbackContext):
        user_id = update.message.from_user.id
        if user_id not in self._whitelist:
            update.message.reply_text(
                "You are not yet on the whitelist. "
                + f"Add {user_id} to your config to receive notifications from me"
            )
            return

        print(update.message.from_user.username, "tries to release some resources")
        to_release = []
        if len(context.args) > 1:
            print("Invalid usage of /rel")
            update.message.reply_text(
                "Usage: /rel <gpu_id> to release <gpu_id> or simply /rel to release all"
            )
            return
        elif len(context.args) == 0:
            to_release = list(range(pynvml.nvmlDeviceGetCount()))
        else:
            gid = int(context.args[0])
            if gid < 0 or gid >= pynvml.nvmlDeviceGetCount():
                update.message.reply_text(
                    f"Invalid GPU ID. Must be between 0 and {pynvml.nvmlDeviceGetCount() - 1}"
                )
            to_release = [gid]

        for gid in to_release:
            if gid not in self.cuda_buffers:
                update.message.reply_text(f"GPU {gid} is not occupied")
            else:
                del self.cuda_buffers[gid]
                cuda.select_device(gid).reset()
                update.message.reply_text(f"Released GPU {gid}")

    def _get_gpu(self, update: Update, context: CallbackContext):
        print(update.message.from_user.username, "requested gpu usage")
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            update.message.reply_text(
                get_usage_msg(i, handle), parse_mode=telegram.ParseMode.MARKDOWN
            )

    def _query_once(self, handles):
        new_states = []
        for handle in handles:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_gb = info.used / (1024**3)
            thresh_gb = 1
            if used_gb > thresh_gb:
                users = [info[0] for info in get_user_info(handle)]
                for i in range(len(users)):
                    if users[i] == "chunqiu2":
                        users[i] += "🐍"
                    elif users[i] == "ywei40":
                        users[i] += "💩"
                    elif users[i] == "yifeng6":
                        users[i] += "🐝"
                    elif users[i] == "shizhuo2":
                        users[i] += "🎡"
                    elif users[i] == "yinlind2":
                        users[i] += "🌲"
                    elif users[i] == "cy54":
                        users[i] += "🎣"
                    else:
                        users[i] += "💸"
                new_states.append(f"🤡OCCUPIED by {', '.join(users)}")
            else:
                new_states.append(f"🚀AVAILABLE")
        msg = f"# New GPU Status at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        for i, state in enumerate(new_states):
            msg += f"* GPU #{i}: {state}\n"
        return new_states, msg

    def _get_gpu_quick(self, update: Update, context: CallbackContext):
        pynvml.nvmlInit()
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(pynvml.nvmlDeviceGetCount())
        ]
        _, msg = self._query_once(handles)
        update.message.reply_text(msg, parse_mode=telegram.ParseMode.MARKDOWN)

    # ! Restricted
    def _poll_gpu(self):
        pynvml.nvmlInit()
        old_states = [None] * pynvml.nvmlDeviceGetCount()
        while True:
            time.sleep(self.interval)
            handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
            try:
                new_states, msg = self._query_once(handles)
            except Exception:
                # reply log
                msg = format_exc()
                print(msg)
                for chat_id in self._whitelist:
                    try:
                        self._updater.bot.send_message(chat_id, msg)
                    except telegram.error.Unauthorized:
                        print("Unauthorized for", chat_id)
                continue

            if new_states != old_states:
                for chat_id in self._whitelist:
                    try:
                        self._updater.bot.send_message(
                            chat_id, msg, parse_mode=telegram.ParseMode.MARKDOWN
                        )
                    except telegram.error.Unauthorized:
                        print("Unauthorized for", chat_id)
                old_states = new_states


if __name__ == "__main__":
    NotifyBot()
