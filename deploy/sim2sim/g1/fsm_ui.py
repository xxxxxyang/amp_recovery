#!/usr/bin/env python3
"""Small local UI for requesting sim2sim FSM states over UDP."""

import argparse
import socket
import tkinter as tk
from tkinter import ttk


STATES = (
    ("Passive\nLT+B", "Passive"),
    ("PD_Stand\nLT+Y", "PDStand"),
    ("Getup\nLT+Up", "Getup"),
)

class FsmUi:
    def __init__(self, root: tk.Tk, host: str, port: int) -> None:
        self.root = root
        self.address = (host, port)
        self.root.title("AMP_mjlab G1 FSM")
        self.root.resizable(False, False)

        style = ttk.Style(root)
        style.configure("TButton", padding=(12, 8))

        frame = ttk.Frame(root, padding=12)
        frame.grid(sticky="nsew")
        ttk.Label(
            frame,
            text="G1 FSM control",
            font=("TkDefaultFont", 12, "bold"),
        ).grid(
            row=0, column=0, columnspan=len(STATES), pady=(0, 8)
        )

        for column, (label, state_name) in enumerate(STATES):
            ttk.Button(
                frame,
                text=label,
                command=lambda name=state_name: self.send_state(name),
                width=12,
            ).grid(row=1, column=column, padx=4, pady=4, sticky="ew")
            frame.columnconfigure(column, weight=1)

        self.status = tk.StringVar(value=f"Ready | {host}:{port}")
        ttk.Label(frame, textvariable=self.status).grid(
            row=2, column=0, columnspan=len(STATES), pady=(8, 0)
        )

    def send_state(self, state_name: str) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
                client.settimeout(0.5)
                client.sendto(state_name.encode("utf-8"), self.address)
            response, _ = client.recvfrom(128)
            message = response.decode("utf-8", errors="replace")
            if message.startswith("queued:"):
                self.status.set(f"Queued: {state_name}")
            else:
                self.status.set(message)
        except OSError as error:
            self.status.set(f"Send failed: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="controller host (default: localhost)")
    parser.add_argument("--port", type=int, default=9876, help="controller UDP port (default: 9876)")
    args = parser.parse_args()

    root = tk.Tk()
    FsmUi(root, args.host, args.port)
    root.mainloop()


if __name__ == "__main__":
    main()
